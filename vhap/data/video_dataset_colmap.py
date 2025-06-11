import os
from pathlib import Path
from copy import deepcopy
from typing import Optional
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, default_collate
from vhap.util import camera
from vhap.util.log import get_logger
from vhap.config.base import DataConfig


logger = get_logger(__name__)


class VideoDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        img_to_tensor: bool = False,
        batchify_all_views: bool = False,
        camera_interval: int = 40,  # Added camera_interval parameter with a default value
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <root_folder>/
                |---images/
                |   |---<camera_id>.jpg
                |
                |---alpha_maps/
                |   |---<camera_id>.png
                |
                |---landmark2d/
                        |---face-alignment/
                        |    |---<camera_id>.npz
                        |
                        |---STAR/
                                |---<camera_id>.npz
            camera_interval: Interval for sampling cameras, determines how many timesteps to create
        """
        super().__init__()
        self.cfg = cfg
        self.img_to_tensor = img_to_tensor
        self.batchify_all_views = batchify_all_views
        self.camera_interval = camera_interval  # Store camera_interval
        # self.global_extrinsic= []

        sequence_paths = self.match_sequences()
        if len(sequence_paths) > 1:
            logger.info(f"Found multiple sequences: {sequence_paths}")
            raise ValueError(
                f"Found multiple sequences by '{cfg.sequence}': \n"
                + "\n\t".join([str(x) for x in sequence_paths])
            )
        elif len(sequence_paths) == 0:
            raise ValueError(f"Cannot find sequence: {cfg.sequence}")
        self.sequence_path = sequence_paths[0]
        logger.info(f"Initializing dataset from {self.sequence_path}")
        self.define_properties()

        # Get camera IDs from image filenames
        self.camera_ids = sorted(
            [
                f.split(".")[0]
                for f in os.listdir(
                    self.sequence_path / self.properties["rgb"]["folder"]
                )
                if f.endswith(self.properties["rgb"]["suffix"])
            ]
        )

        # Create the visual directory if needed
        visual_dir = self.sequence_path / "camera_visual"
        if not visual_dir.exists():
            visual_dir.mkdir(parents=True, exist_ok=True)

        # Initialize timesteps based on camera_interval
        self.initialize_timesteps()

        self.filter_division(cfg.division)
        self.filter_subset(cfg.subset)

        # Distribute cameras across timesteps based on interval
        self.distribute_cameras_by_interval()

        # Load camera parameters for each camera ID
        self.load_camera_params()

        logger.info(
            f"number of timesteps: {self.num_timesteps}, number of cameras: {self.num_cameras}"
        )

        # collect items
        self.items = []
        for ti, timestep_index in enumerate(self.timestep_indices):
            # Get cameras assigned to this timestep
            timestep_cameras = self.timestep_camera_map.get(timestep_index, [])
            for ci, camera_id in enumerate(timestep_cameras):
                self.items.append(
                    {
                        "timestep_index": ti,  # new index after filtering
                        "timestep_index_original": timestep_index,  # original index
                        "timestep_id": str(timestep_index),
                        "camera_index": ci,
                        "camera_id": camera_id,
                    }
                )

    def match_sequences(self):
        logger.info(
            f"Looking for sequence '{self.cfg.sequence}' at {self.cfg.root_folder}"
        )
        return list(
            filter(
                lambda x: x.is_dir(), self.cfg.root_folder.glob(f"{self.cfg.sequence}*")
            )
        )

    def define_properties(self):
        self.properties = {
            "rgb": {
                "folder": (
                    f"images_{self.cfg.n_downsample_rgb}"
                    if self.cfg.n_downsample_rgb
                    else "images"
                ),
                "per_timestep": False,  # Changed to False as we're now using camera_id based paths
                "suffix": "jpg",
            },
            "alpha_map": {
                "folder": "alpha_maps",
                "per_timestep": False,  # Changed to False
                "suffix": "jpg",
            },
            "landmark2d/face-alignment": {
                "folder": "landmark2d/face-alignment",
                "per_timestep": False,
                "suffix": "npz",
            },
            "landmark2d/STAR": {
                "folder": "landmark2d/STAR",
                "per_timestep": False,
                "suffix": "npz",
            },
        }

    @staticmethod
    def get_number_after_prefix(string, prefix):
        i = string.find(prefix)
        if i != -1:
            number_begin = i + len(prefix)
            assert number_begin < len(
                string
            ), f"No number found behind prefix '{prefix}'"
            assert string[
                number_begin
            ].isdigit(), f"No number found behind prefix '{prefix}'"

            non_digit_indices = [
                i for i, c in enumerate(string[number_begin:]) if not c.isdigit()
            ]
            if len(non_digit_indices) > 0:
                number_end = number_begin + min(non_digit_indices)
                return int(string[number_begin:number_end])
            else:
                return int(string[number_begin:])
        else:
            return None

    def filter_division(self, division):
        pass

    def filter_subset(self, subset):
        if subset is not None:
            if "ti" in subset:
                ti = self.get_number_after_prefix(subset, "ti")
                if "tj" in subset:
                    tj = self.get_number_after_prefix(subset, "tj")
                    self.timestep_indices = self.timestep_indices[ti : tj + 1]
                else:
                    self.timestep_indices = self.timestep_indices[ti : ti + 1]
            elif "tn" in subset:
                tn = self.get_number_after_prefix(subset, "tn")
                tn_all = len(self.timestep_indices)
                tn = min(tn, tn_all)
                self.timestep_indices = self.timestep_indices[:: tn_all // tn][:tn]
            elif "ts" in subset:
                ts = self.get_number_after_prefix(subset, "ts")
                self.timestep_indices = self.timestep_indices[::ts]
            if "ci" in subset:
                ci = self.get_number_after_prefix(subset, "ci")
                self.camera_ids = self.camera_ids[ci : ci + 1]
            elif "cn" in subset:
                cn = self.get_number_after_prefix(subset, "cn")
                cn_all = len(self.camera_ids)
                cn = min(cn, cn_all)
                self.camera_ids = self.camera_ids[:: cn_all // cn][:cn]
            elif "cs" in subset:
                cs = self.get_number_after_prefix(subset, "cs")
                self.camera_ids = self.camera_ids[::cs]

    def initialize_timesteps(self):
        """Initialize timestep IDs and indices based on camera interval"""
        if self.camera_interval <= 0:
            # Default: all cameras in one timestep
            self.timestep_ids = ["0"]
            self.timestep_indices = [0]
        else:
            # Create as many timesteps as needed to distribute cameras
            num_timesteps = min(self.camera_interval, len(self.camera_ids))
            self.timestep_ids = [str(i) for i in range(num_timesteps)]
            self.timestep_indices = list(range(num_timesteps))

        logger.info(
            f"Initialized {len(self.timestep_indices)} timesteps based on camera interval {self.camera_interval}"
        )

    def distribute_cameras_by_interval(self):
        """
        Distribute cameras across timesteps based on camera_interval.
        Camera i goes to timestep (i % camera_interval).
        """
        self.timestep_camera_map = {}

        if self.camera_interval <= 0:
            # Default behavior: all cameras in timestep 0
            self.timestep_camera_map[0] = self.camera_ids
            return

        # Distribute cameras by modulo of their index
        for i, camera_id in enumerate(self.camera_ids):
            timestep_idx = i % len(self.timestep_indices)
            if timestep_idx not in self.timestep_camera_map:
                self.timestep_camera_map[timestep_idx] = []
            self.timestep_camera_map[timestep_idx].append(camera_id)

        # Log the distribution
        for timestep_idx in sorted(self.timestep_camera_map.keys()):
            camera_list = self.timestep_camera_map[timestep_idx]
            logger.info(
                f"Timestep {timestep_idx} has {len(camera_list)} cameras: {camera_list[:3]}..."
            )

    def load_camera_params(self):
        cameras_folder = Path("../data/mono/luo0425/cameras")
        print(f"Loading camera parameters from {cameras_folder}")
        # /home/zhangzhh12024/Avatars/data/104_colmap/cameras
        self.camera_params = {}

        # Create a dictionary structure to store camera parameters
        # First by timestep_id, then by camera_id
        for timestep_idx in self.timestep_indices:
            timestep_id = str(timestep_idx)
            self.camera_params[timestep_id] = {}

        # For each camera, load its parameters and store them under the appropriate timestep
        for camera_id in self.camera_ids:
            # The camera file should match the camera_id format
            # Assuming camera_id is already in the correct format for file lookup
            frame_id = camera_id[-4:]  # Use camera_id directly to find the camera file
            camera_file = cameras_folder / frame_id / "camera_00.npz"

            if not camera_file.exists():
                logger.warning(
                    f"Camera file not found for camera_id {camera_id}: {camera_file}"
                )
                continue

            # Load camera parameters from NPZ file
            try:
                camera_data = np.load(camera_file)

                # Convert numpy arrays to tensors
                intrinsic = torch.from_numpy(camera_data["intrinsic.npy"]).float()
                extrinsic = torch.from_numpy(camera_data["extrinsic.npy"]).float()

                # extrinsic[:3, 3] *= 0.1
                # flip = False
                # if flip:

                #     scale_matrix = torch.eye(4)
                #     scale_matrix[1, 1] = -1  # Flip Y axis
                #     scale_matrix[2, 2] = -1  # Flip Z axis
                #     # Apply transformation
                #     # We need to create a 4x4 matrix first from the 3x4 extrinsic
                #     extrinsic_4x4 = torch.eye(4)
                #     extrinsic_4x4[:3, :4] = extrinsic
                #     # Apply the transformation: first scale, then the extrinsic
                #     extrinsic_4x4 = extrinsic_4x4 @ scale_matrix
                #     # Extract back the 3x4 matrix
                #     extrinsic = extrinsic_4x4[:3, :4]

                if camera_id == self.camera_ids[0]:
                    self.global_extrinsic = extrinsic.clone()
                    print(f"global_extrinsic: {self.global_extrinsic}")

                # Find which timestep this camera belongs to
                for timestep_idx, cameras in self.timestep_camera_map.items():
                    if camera_id in cameras:
                        timestep_id = str(timestep_idx)
                        # Store camera parameters under the appropriate timestep and camera ID
                        self.camera_params[timestep_id][camera_id] = {
                            "intrinsic": intrinsic,
                            "extrinsic": extrinsic,
                        }
                        break

            except Exception as e:
                logger.error(f"Error loading camera file {camera_file}: {e}")

        # Generate visualization for each timestep
        for timestep_idx in self.timestep_indices:
            timestep_id = str(timestep_idx)
            # Skip if no cameras for this timestep
            if not self.camera_params[timestep_id]:
                continue

            # Get the first camera in this timestep for reference
            first_camera_id = next(iter(self.camera_params[timestep_id]))
            first_camera = self.camera_params[timestep_id][first_camera_id]

            # Save the cameras in this timestep as OBJ
            obj_filename = f"camera_visual/cameras_timestep_{timestep_id}.obj"
            self.save_cameras_as_obj(
                first_camera["extrinsic"],
                first_camera["intrinsic"],
                timestep_id,
                obj_filename,
            )

        # Create a combined visualization with global extrinsic and model initial position
        self.save_combined_visualization()

        return self.camera_params

    def save_combined_visualization(self):
        """
        Creates a combined OBJ file with:
        - Coordinate axes
        - Global extrinsic camera frustum
        - Initial model position and rotation (represented as a simple arrow)
        """
        obj_path = self.sequence_path / "camera_visual" / "combined_visualization.obj"
        mtl_path = obj_path.with_suffix(".mtl")

        logger.info(f"Creating combined visualization at {obj_path}")

        # Create material file
        with open(mtl_path, "w") as mtl:
            mtl.write("# Combined visualization materials\n")

            # Reference plane material
            mtl.write("newmtl reference_plane\n")
            mtl.write("Kd 0.8 0.8 0.8\n")  # Light gray
            mtl.write("Ka 0.1 0.1 0.1\n")
            mtl.write("d 0.3\n\n")  # 30% transparency

            # Coordinate axes materials
            mtl.write("newmtl axis_x\n")
            mtl.write("Kd 1.0 0.0 0.0\n")  # Red for X axis
            mtl.write("Ka 0.2 0.0 0.0\n\n")

            mtl.write("newmtl axis_y\n")
            mtl.write("Kd 0.0 1.0 0.0\n")  # Green for Y axis
            mtl.write("Ka 0.0 0.2 0.0\n\n")

            mtl.write("newmtl axis_z\n")
            mtl.write("Kd 0.0 0.0 1.0\n")  # Blue for Z axis
            mtl.write("Ka 0.0 0.0 0.2\n\n")

            # Global camera frustum
            mtl.write("newmtl global_camera\n")
            mtl.write("Kd 0.8 0.2 0.8\n")  # Purple for global camera
            mtl.write("Ka 0.2 0.05 0.2\n\n")

            # Initial model position
            mtl.write("newmtl model_position\n")
            mtl.write("Kd 1.0 0.8 0.0\n")  # Gold for model
            mtl.write("Ka 0.3 0.2 0.0\n\n")

            # Initial model direction
            mtl.write("newmtl model_direction\n")
            mtl.write("Kd 0.0 0.8 0.8\n")  # Cyan for model direction
            mtl.write("Ka 0.0 0.2 0.2\n\n")

        with open(obj_path, "w") as f:
            f.write(
                "# Combined visualization with global extrinsic and model position\n"
            )
            f.write(f"mtllib {mtl_path.name}\n\n")

            # Initialize vertex counter
            self.vertex_count = 1

            # Create coordinate axes at origin
            self.vertex_count = self.create_axis_visualization(f, self.vertex_count)

            # Define reference plane (XY-plane at Z=0)
            plane_size = 2.0
            plane_vertices = [
                (-plane_size, -plane_size, 0.0),  # bottom-left
                (plane_size, -plane_size, 0.0),  # bottom-right
                (plane_size, plane_size, 0.0),  # top-right
                (-plane_size, plane_size, 0.0),  # top-left
            ]

            # Write reference plane vertices
            f.write("# Reference plane vertices\n")
            for v in plane_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write reference plane face with material
            f.write("\n# Reference plane face\n")
            f.write("usemtl reference_plane\n")
            f.write(
                f"f {self.vertex_count} {self.vertex_count+1} {self.vertex_count+2} {self.vertex_count+3}\n\n"
            )

            # Update vertex counter
            self.vertex_count += 4

            # Add global extrinsic camera visualization
            if hasattr(self, "global_extrinsic"):
                f.write("# Global Extrinsic Camera\n")
                f.write("usemtl global_camera\n")

                # Create a 4x4 matrix for the global extrinsic
                extrinsic_matrix = torch.eye(4)
                extrinsic_matrix[:3, :4] = self.global_extrinsic

                # Create camera frustum with larger scale for visibility
                intrinsic_matrix = torch.eye(
                    3
                )  # Use identity for visualization purpose
                intrinsic_matrix[0, 0] = intrinsic_matrix[1, 1] = (
                    1000  # Just for visualization
                )

                # Draw the camera frustum
                self.create_camera_frustum(
                    extrinsic_matrix.numpy(),
                    intrinsic_matrix.numpy(),
                    [0.8, 0.2, 0.8],  # Purple color
                    f,
                    scale=0.5,  # Larger scale for better visibility
                )

            # Add initial model position and direction
            try:
                # Calculate where the model would be placed using camera information
                if hasattr(self, "global_extrinsic"):
                    # Extract rotation and translation from global extrinsic
                    rotation = self.global_extrinsic[:3, :3].numpy()
                    translation = self.global_extrinsic[:3, 3].numpy()

                    # The camera position in world space
                    camera_position = -np.matmul(rotation.T, translation)

                    # Forward direction from camera space to world space
                    forward_direction = np.matmul(rotation.T, np.array([0, 0, -1.0]))
                    forward_direction = forward_direction / np.linalg.norm(
                        forward_direction
                    )
                    distance = -1
                    # Model position at 1 unit distance from camera
                    initial_position = camera_position + distance * forward_direction

                    # Calculate model facing direction (negative of forward direction)
                    model_direction = -forward_direction

                    # Add model position marker (a sphere represented by a point)
                    f.write("\n# Initial Model Position\n")
                    f.write("usemtl model_position\n")

                    # Define a simple sphere at the initial position
                    sphere_radius = 0.1
                    sphere_segments = 8
                    sphere_rings = 8

                    # Vertices for the sphere
                    sphere_vertices = []
                    for i in range(sphere_rings + 1):
                        phi = np.pi * i / sphere_rings
                        for j in range(sphere_segments):
                            theta = 2 * np.pi * j / sphere_segments
                            x = initial_position[0] + sphere_radius * np.sin(
                                phi
                            ) * np.cos(theta)
                            y = initial_position[1] + sphere_radius * np.sin(
                                phi
                            ) * np.sin(theta)
                            z = initial_position[2] + sphere_radius * np.cos(phi)
                            sphere_vertices.append((x, y, z))
                            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

                    # Faces for the sphere
                    for i in range(sphere_rings):
                        for j in range(sphere_segments):
                            p1 = self.vertex_count + i * sphere_segments + j
                            p2 = (
                                self.vertex_count
                                + i * sphere_segments
                                + (j + 1) % sphere_segments
                            )
                            p3 = (
                                self.vertex_count
                                + (i + 1) * sphere_segments
                                + (j + 1) % sphere_segments
                            )
                            p4 = self.vertex_count + (i + 1) * sphere_segments + j
                            f.write(f"f {p1} {p2} {p3} {p4}\n")

                    self.vertex_count += len(sphere_vertices)

                    # Add an arrow showing initial model direction
                    f.write("\n# Initial Model Direction\n")
                    f.write("usemtl model_direction\n")

                    # Arrow starts at initial position and points in model_direction
                    arrow_length = 0.3
                    arrow_end = initial_position + arrow_length * model_direction

                    # Arrow shaft
                    f.write(
                        f"v {initial_position[0]:.6f} {initial_position[1]:.6f} {initial_position[2]:.6f}\n"
                    )
                    f.write(
                        f"v {arrow_end[0]:.6f} {arrow_end[1]:.6f} {arrow_end[2]:.6f}\n"
                    )
                    f.write(f"l {self.vertex_count} {self.vertex_count+1}\n")
                    self.vertex_count += 2

                    # Arrow head (slightly thicker)
                    # Find perpendicular vectors for the arrow head
                    if np.abs(model_direction[0]) < np.abs(model_direction[1]):
                        perp1 = np.array([0, model_direction[2], -model_direction[1]])
                    else:
                        perp1 = np.array([model_direction[2], 0, -model_direction[0]])
                    perp1 = perp1 / np.linalg.norm(perp1)
                    perp2 = np.cross(model_direction, perp1)

                    # Create cone at the arrow head
                    arrow_head_radius = 0.05
                    segments = 8
                    for i in range(segments):
                        angle = 2 * np.pi * i / segments
                        # Point on the circle at the base of the cone
                        x = (
                            arrow_end[0]
                            - 0.1 * model_direction[0]
                            + arrow_head_radius
                            * (perp1[0] * np.cos(angle) + perp2[0] * np.sin(angle))
                        )
                        y = (
                            arrow_end[1]
                            - 0.1 * model_direction[1]
                            + arrow_head_radius
                            * (perp1[1] * np.cos(angle) + perp2[1] * np.sin(angle))
                        )
                        z = (
                            arrow_end[2]
                            - 0.1 * model_direction[2]
                            + arrow_head_radius
                            * (perp1[2] * np.cos(angle) + perp2[2] * np.sin(angle))
                        )
                        f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

                    # Cone apex (arrow head tip)
                    f.write(
                        f"v {arrow_end[0]:.6f} {arrow_end[1]:.6f} {arrow_end[2]:.6f}\n"
                    )
                    cone_apex = self.vertex_count + segments

                    # Faces connecting the base circle to the apex
                    for i in range(segments):
                        p1 = self.vertex_count + i
                        p2 = self.vertex_count + (i + 1) % segments
                        f.write(f"f {p1} {p2} {cone_apex}\n")

                    self.vertex_count += segments + 1

            except Exception as e:
                logger.error(f"Error creating model position visualization: {e}")

        logger.info(f"Combined visualization saved to {obj_path}")

    def save_cameras_as_obj(self, extrinsic, intrinsic, timestep_id, obj_filename=None):
        """
        Save camera parameters as an OBJ file.

        Args:
            extrinsic: Camera extrinsic matrix (reference)
            intrinsic: Camera intrinsic matrix (reference)
            timestep_id: The timestep ID to visualize
            obj_filename: Name for the OBJ file (default: camera_visualization.obj)
        """
        logger.info(f"Saving camera parameters for timestep {timestep_id} as OBJ file")

        if obj_filename is None:
            obj_path = self.sequence_path / "camera_visual" / "camera_visualization.obj"
        else:
            obj_path = self.sequence_path / obj_filename

        mtl_path = obj_path.with_suffix(".mtl")

        # Create material file with colors from red to blue
        with open(mtl_path, "w") as mtl:
            mtl.write("# Camera visualization materials\n")
            mtl.write("newmtl reference_plane\n")
            mtl.write("Kd 0.8 0.8 0.8\n")  # Light gray
            mtl.write("Ka 0.1 0.1 0.1\n")
            mtl.write("d 0.3\n\n")  # 30% transparency

            # Coordinate axes materials
            mtl.write("newmtl axis_x\n")
            mtl.write("Kd 1.0 0.0 0.0\n")  # Red for X axis
            mtl.write("Ka 0.2 0.0 0.0\n\n")

            mtl.write("newmtl axis_y\n")
            mtl.write("Kd 0.0 1.0 0.0\n")  # Green for Y axis
            mtl.write("Ka 0.0 0.2 0.0\n\n")

            mtl.write("newmtl axis_z\n")
            mtl.write("Kd 0.0 0.0 1.0\n")  # Blue for Z axis
            mtl.write("Ka 0.0 0.0 0.2\n\n")

            # Camera color materials
            cameras = self.timestep_camera_map.get(int(timestep_id), [])
            total_cameras = len(cameras)

            for i, camera_id in enumerate(cameras):
                # Create gradient from red (1,0,0) to blue (0,0,1)
                r = 1.0 - (i / (total_cameras - 1) if total_cameras > 1 else 0)
                b = i / (total_cameras - 1) if total_cameras > 1 else 0
                g = 0.2  # Small green component for better visualization

                color_str = f"{r:.3f}_{g:.3f}_{b:.3f}"
                mtl.write(f"newmtl camera_{color_str}\n")
                mtl.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
                mtl.write(f"Ka {r/4:.4f} {g/4:.4f} {b/4:.4f}\n\n")

        with open(obj_path, "w") as f:
            f.write("# Camera visualization\n")
            f.write("# Each camera is represented by a frustum\n")
            f.write(f"mtllib {mtl_path.name}\n\n")

            # Initialize vertex counter
            self.vertex_count = 1

            # Create coordinate axes at origin
            self.vertex_count = self.create_axis_visualization(f, self.vertex_count)

            # Define reference plane (XY-plane at Z=0)
            plane_size = 2.0
            plane_vertices = [
                (-plane_size, -plane_size, 0.0),  # bottom-left
                (plane_size, -plane_size, 0.0),  # bottom-right
                (plane_size, plane_size, 0.0),  # top-right
                (-plane_size, plane_size, 0.0),  # top-left
            ]

            # Write reference plane vertices
            f.write("# Reference plane vertices\n")
            for v in plane_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write reference plane face with material
            f.write("\n# Reference plane face\n")
            f.write("usemtl reference_plane\n")
            f.write(
                f"f {self.vertex_count} {self.vertex_count+1} {self.vertex_count+2} {self.vertex_count+3}\n\n"
            )

            # Update vertex counter
            self.vertex_count += 4

            # Get cameras for this timestep
            cameras = self.timestep_camera_map.get(int(timestep_id), [])

            # For each camera in this timestep
            for i, camera_id in enumerate(cameras):
                # Get camera parameters
                if camera_id not in self.camera_params[timestep_id]:
                    continue

                camera_param = self.camera_params[timestep_id][camera_id]

                # Get camera extrinsic in world-to-camera format
                if self.cfg.target_extrinsic_type == "w2c":
                    # Already in w2c format
                    extrinsic_w2c = camera_param["extrinsic"]
                    # Create a 4x4 matrix
                    extrinsic_matrix = np.eye(4)
                    extrinsic_matrix[:3, :4] = extrinsic_w2c.numpy()
                else:
                    # Convert c2w to w2c
                    extrinsic_c2w = camera_param["extrinsic"].numpy()
                    # Create a 4x4 matrix
                    c2w_matrix = np.eye(4)
                    c2w_matrix[:3, :4] = extrinsic_c2w
                    # Invert to get w2c
                    extrinsic_matrix = np.linalg.inv(c2w_matrix)

                # Get intrinsic matrix
                intrinsic_matrix = camera_param["intrinsic"].numpy()

                # Create color for this camera (red to blue gradient)
                r = 1.0 - (i / (len(cameras) - 1) if len(cameras) > 1 else 0)
                b = i / (len(cameras) - 1) if len(cameras) > 1 else 0
                g = 0.2
                color = [r, g, b]

                # Add camera frustum to the OBJ file
                f.write(f"\n# Camera frustum for camera {camera_id}\n")
                self.create_camera_frustum(
                    extrinsic_matrix, intrinsic_matrix, color, f, scale=0.2
                )

        logger.info(
            f"Camera visualization saved to {obj_path} with materials in {mtl_path}"
        )

    def create_camera_frustum(self, extrinsic, intrinsic, color, f, scale=0.2):
        """
        Create a camera frustum in the OBJ file
        Args:
            extrinsic: 4x4 camera extrinsic matrix (world to camera)
            intrinsic: 3x3 camera intrinsic matrix
            color: RGB color for the camera frustum (0-1 range)
            f: File object to write to
            scale: Scale factor for the frustum size
        Returns:
            Updated vertex count
        """
        # Camera center is the translation part of the inverse extrinsic matrix
        camera_to_world = np.linalg.inv(extrinsic)
        cam_center = camera_to_world[:3, 3]

        # Camera rotation from world coordinates
        rotation = camera_to_world[:3, :3]

        # Create frustum points
        frustum_points = []

        # Camera center is at the apex of the pyramid
        frustum_points.append(cam_center)

        # Create corners of the frustum at a fixed distance
        distance = scale

        # Get aspect ratio from intrinsics
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        aspect_ratio = fx / fy

        # Calculate the size of the frustum
        half_height = distance * 0.5
        half_width = half_height * aspect_ratio

        # Camera coordinate system vectors (in world space)
        forward = rotation @ np.array(
            [0, 0, 1]
        )  # Z axis points forward in camera space
        up = rotation @ np.array([0, 1, 0])  # Y axis points up in camera space
        right = rotation @ np.array([1, 0, 0])  # X axis points right in camera space

        # Create the four corners of the frustum plane
        # Top-left, top-right, bottom-right, bottom-left
        corners = [
            cam_center + distance * forward - half_width * right + half_height * up,
            cam_center + distance * forward + half_width * right + half_height * up,
            cam_center + distance * forward + half_width * right - half_height * up,
            cam_center + distance * forward - half_width * right - half_height * up,
        ]
        frustum_points.extend(corners)

        # Write material with the specified color
        f.write(f"usemtl camera_{color[0]:.3f}_{color[1]:.3f}_{color[2]:.3f}\n")

        # Write vertices
        for point in frustum_points:
            f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

        # Write faces - connecting camera center to each corner
        start_idx = self.vertex_count

        # Draw the four triangular faces of the pyramid
        for i in range(4):
            next_i = (i + 1) % 4
            f.write(f"f {start_idx} {start_idx+1+i} {start_idx+1+next_i}\n")

        # Draw the base (far plane) of the frustum
        f.write(f"f {start_idx+1} {start_idx+2} {start_idx+3} {start_idx+4}\n")

        # Draw edges
        f.write(f"l {start_idx} {start_idx+1}\n")
        f.write(f"l {start_idx} {start_idx+2}\n")
        f.write(f"l {start_idx} {start_idx+3}\n")
        f.write(f"l {start_idx} {start_idx+4}\n")

        # Draw base edges
        f.write(f"l {start_idx+1} {start_idx+2}\n")
        f.write(f"l {start_idx+2} {start_idx+3}\n")
        f.write(f"l {start_idx+3} {start_idx+4}\n")
        f.write(f"l {start_idx+4} {start_idx+1}\n")

        # Update vertex count
        self.vertex_count += 5  # camera center + 4 corners
        return self.vertex_count

    def create_axis_visualization(self, f, vertex_count):
        """
        Create a better visualization for coordinate axes with labels

        Args:
            f: File object to write to
            vertex_count: Current vertex counter

        Returns:
            Updated vertex count
        """
        # Axis parameters
        axis_length = 1.0
        thickness = 0.02  # Thickness of the axes

        # Write axis vertices
        f.write("# Coordinate axes with clear visualization\n")

        # Create XYZ axes with thickness (using boxes for each axis)
        # X axis (red)
        f.write("usemtl axis_x\n")
        x_vertices = [
            (0, -thickness, -thickness),  # base face
            (0, thickness, -thickness),
            (0, thickness, thickness),
            (0, -thickness, thickness),
            (axis_length, -thickness, -thickness),  # end face
            (axis_length, thickness, -thickness),
            (axis_length, thickness, thickness),
            (axis_length, -thickness, thickness),
        ]
        for v in x_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # X axis faces (6 faces of the box)
        start_idx = vertex_count
        # end face
        f.write(f"f {start_idx+4} {start_idx+5} {start_idx+6} {start_idx+7}\n")
        # side faces
        f.write(f"f {start_idx} {start_idx+4} {start_idx+7} {start_idx+3}\n")
        f.write(f"f {start_idx+1} {start_idx+5} {start_idx+4} {start_idx}\n")
        f.write(f"f {start_idx+2} {start_idx+6} {start_idx+5} {start_idx+1}\n")
        f.write(f"f {start_idx+3} {start_idx+7} {start_idx+6} {start_idx+2}\n")

        # Create X label
        f.write("# X label\n")
        x_label_pos = axis_length + 0.1
        # Simple X shape using lines
        x_label_size = 0.05
        x_label_vertices = [
            (x_label_pos, -x_label_size, -x_label_size),  # bottom-left to top-right
            (x_label_pos, x_label_size, x_label_size),
            (x_label_pos, -x_label_size, x_label_size),  # bottom-right to top-left
            (x_label_pos, x_label_size, -x_label_size),
        ]
        for v in x_label_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write(f"l {start_idx+8} {start_idx+9}\n")
        f.write(f"l {start_idx+10} {start_idx+11}\n")

        vertex_count += 12  # 8 for X axis + 4 for X label

        # Y axis (green)
        f.write("usemtl axis_y\n")
        y_vertices = [
            (-thickness, 0, -thickness),  # base face
            (thickness, 0, -thickness),
            (thickness, 0, thickness),
            (-thickness, 0, thickness),
            (-thickness, axis_length, -thickness),  # end face
            (thickness, axis_length, -thickness),
            (thickness, axis_length, thickness),
            (-thickness, axis_length, thickness),
        ]
        for v in y_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Y axis faces
        start_idx = vertex_count
        # end face
        f.write(f"f {start_idx+4} {start_idx+5} {start_idx+6} {start_idx+7}\n")
        # side faces
        f.write(f"f {start_idx} {start_idx+4} {start_idx+7} {start_idx+3}\n")
        f.write(f"f {start_idx+1} {start_idx+5} {start_idx+4} {start_idx}\n")
        f.write(f"f {start_idx+2} {start_idx+6} {start_idx+5} {start_idx+1}\n")
        f.write(f"f {start_idx+3} {start_idx+7} {start_idx+6} {start_idx+2}\n")

        # Create Y label
        f.write("# Y label\n")
        y_label_pos = axis_length + 0.1
        y_label_size = 0.05
        y_label_vertices = [
            (0, y_label_pos, 0),  # center
            (-y_label_size, y_label_pos + y_label_size, 0),  # top vertices
            (y_label_size, y_label_pos + y_label_size, 0),
            (0, y_label_pos - y_label_size, 0),  # bottom
        ]
        for v in y_label_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write(f"l {start_idx+8} {start_idx+9}\n")
        f.write(f"l {start_idx+8} {start_idx+10}\n")
        f.write(f"l {start_idx+8} {start_idx+11}\n")

        vertex_count += 12  # 8 for Y axis + 4 for Y label

        # Z axis (blue)
        f.write("usemtl axis_z\n")
        z_vertices = [
            (-thickness, -thickness, 0),  # base face
            (thickness, -thickness, 0),
            (thickness, thickness, 0),
            (-thickness, thickness, 0),
            (-thickness, -thickness, axis_length),  # end face
            (thickness, -thickness, axis_length),
            (thickness, thickness, axis_length),
            (-thickness, thickness, axis_length),
        ]
        for v in z_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Z axis faces
        start_idx = vertex_count
        # end face
        f.write(f"f {start_idx+4} {start_idx+5} {start_idx+6} {start_idx+7}\n")
        # side faces
        f.write(f"f {start_idx} {start_idx+4} {start_idx+7} {start_idx+3}\n")
        f.write(f"f {start_idx+1} {start_idx+5} {start_idx+4} {start_idx}\n")
        f.write(f"f {start_idx+2} {start_idx+6} {start_idx+5} {start_idx+1}\n")
        f.write(f"f {start_idx+3} {start_idx+7} {start_idx+6} {start_idx+2}\n")

        # Create Z label
        f.write("# Z label\n")
        z_label_pos = axis_length + 0.1
        z_label_size = 0.05
        z_label_vertices = [
            (-z_label_size, -z_label_size, z_label_pos),  # top horizontal
            (z_label_size, -z_label_size, z_label_pos),
            (-z_label_size, z_label_size, z_label_pos),  # diagonal
            (z_label_size, -z_label_size, z_label_pos),
            (-z_label_size, z_label_size, z_label_pos),  # bottom horizontal
            (z_label_size, z_label_size, z_label_pos),
        ]
        for v in z_label_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write(f"l {start_idx+8} {start_idx+9}\n")
        f.write(f"l {start_idx+10} {start_idx+11}\n")
        f.write(f"l {start_idx+12} {start_idx+13}\n")

        vertex_count += 14  # 8 for Z axis + 6 for Z label

        return vertex_count

    def __len__(self):
        if self.batchify_all_views:
            return self.num_timesteps
        else:
            return len(self.items)

    def __getitem__(self, i):
        if self.batchify_all_views:
            return self.getitem_by_timestep(i)
        else:
            return self.getitem_single_image(i)

    def getitem_single_image(self, i):
        item = deepcopy(self.items[i])

        rgb_path = self.get_property_path("rgb", i)
        item["rgb"] = np.array(Image.open(rgb_path))

        # Get camera parameters for the camera_id
        timestep_id = item["timestep_id"]
        camera_id = item["camera_id"]
        camera_param = self.camera_params[timestep_id][camera_id].copy()

        item["intrinsic"] = camera_param["intrinsic"].clone()
        item["extrinsic"] = camera_param["extrinsic"].clone()

        if self.cfg.use_alpha_map or self.cfg.background_color is not None:
            alpha_path = self.get_property_path("alpha_map", i)
            item["alpha_map"] = np.array(Image.open(alpha_path))

        if self.cfg.use_landmark:
            timestep_index = self.items[i]["timestep_index"]

            if self.cfg.landmark_source == "face-alignment":
                landmark_path = self.get_property_path("landmark2d/face-alignment", i)
            elif self.cfg.landmark_source == "star":
                landmark_path = self.get_property_path("landmark2d/STAR", i)
            else:
                raise NotImplementedError(
                    f"Unknown landmark source: {self.cfg.landmark_source}"
                )
            landmark_npz = np.load(landmark_path)

            item["lmk2d"] = landmark_npz["face_landmark_2d"][
                timestep_index
            ]  # (num_points, 3)
            if (item["lmk2d"][:, :2] == -1).sum() > 0:
                item["lmk2d"][:, 2:] = 0.0
            else:
                item["lmk2d"][:, 2:] = 1.0

        item = self.apply_transforms(item)
        return item

    def getitem_by_timestep(self, timestep_index):
        """
        Get all items for a specific timestep index
        """
        # Find all items for this timestep
        indices = [
            i
            for i, item in enumerate(self.items)
            if item["timestep_index"] == timestep_index
        ]

        if not indices:
            raise IndexError(f"No items found for timestep index {timestep_index}")

        # Collect all items for this timestep
        item = default_collate([self.getitem_single_image(i) for i in indices])
        item["num_cameras"] = len(indices)
        return item

    def apply_transforms(self, item):
        item = self.apply_scale_factor(item)
        item = self.apply_background_color(item)
        item = self.apply_to_tensor(item)
        return item

    def apply_to_tensor(self, item):
        if self.img_to_tensor:
            if "rgb" in item:
                item["rgb"] = F.to_tensor(item["rgb"])

            if "alpha_map" in item:
                item["alpha_map"] = F.to_tensor(item["alpha_map"])
        return item

    def apply_scale_factor(self, item):
        assert self.cfg.scale_factor <= 1.0

        if "rgb" in item:
            H, W, _ = item["rgb"].shape
            h, w = int(H * self.cfg.scale_factor), int(W * self.cfg.scale_factor)
            rgb = Image.fromarray(item["rgb"]).resize((w, h), resample=Image.BILINEAR)
            item["rgb"] = np.array(rgb)

        # properties that are defined based on image size
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= w
            item["lmk2d"][..., 1] *= h

        if "lmk2d_iris" in item:
            item["lmk2d_iris"][..., 0] *= w
            item["lmk2d_iris"][..., 1] *= h

        if "bbox_2d" in item:
            item["bbox_2d"][[0, 2]] *= w
            item["bbox_2d"][[1, 3]] *= h

        # properties need to be scaled down when rgb is downsampled
        n_downsample_rgb = self.cfg.n_downsample_rgb if self.cfg.n_downsample_rgb else 1
        scale_factor = self.cfg.scale_factor / n_downsample_rgb
        item["scale_factor"] = scale_factor  # NOTE: not self.cfg.scale_factor
        if scale_factor < 1.0:
            if "intrinsic" in item:
                item["intrinsic"][:2] *= scale_factor
            if "alpha_map" in item:
                h, w = item["rgb"].shape[:2]
                alpha_map = Image.fromarray(item["alpha_map"]).resize(
                    (w, h), Image.Resampling.BILINEAR
                )
                item["alpha_map"] = np.array(alpha_map)
        return item

    def apply_background_color(self, item):
        if self.cfg.background_color is not None:
            assert (
                "alpha_map" in item
            ), "'alpha_map' is required to apply background color."
            fg = item["rgb"]
            if self.cfg.background_color == "white":
                bg = np.ones_like(fg) * 255
            elif self.cfg.background_color == "black":
                bg = np.zeros_like(fg)
            else:
                raise NotImplementedError(
                    f"Unknown background color: {self.cfg.background_color}."
                )

            w = item["alpha_map"][..., None] / 255
            if w.ndim == 4:
                w = np.squeeze(w, axis=-1)  # 从 (H, W, C, 1) → (H, W, C)
            img = (w * fg + (1 - w) * bg).astype(np.uint8)
            item["rgb"] = img
        return item

    def get_property_path(
        self,
        name,
        index: Optional[int] = None,
        timestep_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        p = self.properties[name]
        folder = p["folder"] if "folder" in p else None
        per_timestep = p["per_timestep"]
        suffix = p["suffix"]

        path = self.sequence_path
        if folder is not None:
            path = path / folder
        # print(f"get_property_path: {path}")
        # Get camera_id from index if not provided
        # print(f"index: {index}, camera_id {camera_id}")
        if camera_id is None:
            assert (
                index is not None
            ), "index is required when camera_id is not provided."
            camera_id = self.items[index]["camera_id"]

        if "cam_id_prefix" in p:
            camera_id = p["cam_id_prefix"] + camera_id

        # Build the path
        path /= f"{camera_id}.{suffix}"

        return path

    def get_property_path_list(self, name):
        paths = []
        for i in range(len(self.items)):
            img_path = self.get_property_path(name, i)
            paths.append(img_path)
        return paths

    @property
    def num_timesteps(self):
        return len(self.timestep_indices)

    @property
    def num_cameras(self):
        return len(self.camera_ids)


if __name__ == "__main__":
    import tyro
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from vhap.config.base import DataConfig, import_module

    cfg = tyro.cli(DataConfig)
    cfg.use_landmark = False
    dataset = import_module(cfg._target)(
        cfg=cfg,
        img_to_tensor=False,
        batchify_all_views=True,
    )

    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())
    print(sample["rgb"].shape)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    for item in tqdm(dataloader):
        pass

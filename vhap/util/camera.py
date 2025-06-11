#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#


from typing import Tuple, Literal
import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy.spatial.transform import Rotation


def align_cameras_to_axes(
    R: torch.Tensor,
    T: torch.Tensor,
    target_convention: Literal["opengl", "opencv"] = None,
):
    """align the averaged axes of cameras with the world axes.

    Args:
        R: rotation matrix (N, 3, 3)
        T: translation vector (N, 3)
    """
    # The column vectors of R are the basis vectors of each camera.
    # We construct new bases by taking the mean directions of axes, then use Gram-Schmidt
    # process to make them orthonormal
    bases_c2w = gram_schmidt_orthogonalization(R.mean(0))
    if target_convention == "opengl":
        bases_c2w[:, [1, 2]] *= -1  # flip y and z axes
    elif target_convention == "opencv":
        pass
    bases_w2c = bases_c2w.t()

    # convert the camera poses into the new coordinate system
    R = bases_w2c[None, ...] @ R
    T = bases_w2c[None, ...] @ T
    return R, T


def convert_camera_convention(camera_convention_conversion: str, R: torch.Tensor, K: torch.Tensor, H: int, W: int):
    if camera_convention_conversion is not None:
        if camera_convention_conversion == "opencv->opengl":
            R[:, :3, [1, 2]] *= -1
            # flip y of the principal point
            K[..., 1, 2] = H - K[..., 1, 2]
        elif camera_convention_conversion == "opencv->pytorch3d":
            R[:, :3, [0, 1]] *= -1
            # flip x and y of the principal point
            K[..., 0, 2] = W - K[..., 0, 2]
            K[..., 1, 2] = H - K[..., 1, 2]
        elif camera_convention_conversion == "opengl->pytorch3d":
            R[:, :3, [0, 2]] *= -1
            # flip x of the principal point
            K[..., 0, 2] = W - K[..., 0, 2]
        else:
            raise ValueError(
                f"Unknown camera coordinate conversion: {camera_convention_conversion}."
            )
    return R, K


def gram_schmidt_orthogonalization(M: torch.tensor):
    """conducting Gram-Schmidt process to transform column vectors into orthogonal bases

    Args:
        M: An matrix (num_rows, num_cols)
    Return:
        M: An matrix with orthonormal column vectors (num_rows, num_cols)
    """
    num_rows, num_cols = M.shape
    for c in range(1, num_cols):
        M[:, [c - 1, c]] = F.normalize(M[:, [c - 1, c]], p=2, dim=0)
        M[:, [c]] -= M[:, :c] @ (M[:, :c].T @ M[:, [c]])

    M[:, -1] = F.normalize(M[:, -1], p=2, dim=0)
    return M


def projection_from_intrinsics(K: np.ndarray, image_size: Tuple[int], near: float=0.01, far:float=10, flip_y: bool=False, z_sign=-1):
    """
    Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: down, z: in)
    Args:
        K: Intrinsic matrix, (N, 3, 3)
            K = [[
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1],
                ]
            ]
        image_size: (height, width)
    Output:
        proj = [[
                [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                [0.0,    0.0,     z_sign*(far+near) / (far-near), -2*far*near / (far-near)],
                [0.0,    0.0,     z_sign,                     0.0                     ]
            ]
        ]
    """

    B = K.shape[0]
    h, w = image_size

    if K.shape[-2:] == (3, 3):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
    elif K.shape[-1] == 4:
        # fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        fx = K[..., [0]]
        fy = K[..., [1]]
        cx = K[..., [2]]
        cy = K[..., [3]]
    else:
        raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    proj = np.zeros([B, 4, 4])
    proj[:, 0, 0]  = fx * 2 / w 
    proj[:, 1, 1]  = fy * 2 / h
    proj[:, 0, 2]  = (w - 2 * cx) / w
    proj[:, 1, 2]  = (h - 2 * cy) / h
    proj[:, 2, 2]  = z_sign * (far+near) / (far-near)
    proj[:, 2, 3]  = -2*far*near / (far-near)
    proj[:, 3, 2]  = z_sign

    if flip_y:
        proj[:, 1, 1] *= -1
    return proj


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, znear=1e-8, zfar=10, convention: Literal["opengl", "opencv"]="opengl"):
        self.image_width = W
        self.image_height = H
        self.radius_default = r
        self.fovy_default = fovy
        self.znear = znear
        self.zfar = zfar
        self.convention = convention

        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.reset()

    def reset(self):
        """ The internal state of the camera is based on the OpenGL convention, but 
            properties are converted to the target convention when queried.
        """
        self.rot = Rotation.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # OpenGL convention
        self.look_at = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.radius = self.radius_default  # camera distance from center
        self.fovy = self.fovy_default
        if self.convention == "opencv":
            self.z_sign = 1
            self.y_sign = 1
        elif self.convention == "opengl":
            self.z_sign = -1
            self.y_sign = -1
        else:
            raise ValueError(f"Unknown convention: {self.convention}")

    @property
    def fovx(self):
        return self.fovy / self.image_height * self.image_width

    @property
    def intrinsics(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.image_width // 2, self.image_height // 2])

    @property
    def projection_matrix(self):
        return projection_from_intrinsics(self.intrinsics[None], (self.image_height, self.image_width), self.znear, self.zfar, z_sign=self.z_sign)[0]

    @property
    def world_view_transform(self):
        return np.linalg.inv(self.pose)  # world2cam

    @property
    def full_proj_transform(self):
        return self.projection_matrix @ self.world_view_transform

    @property
    def pose(self):
        # first move camera to radius
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] += self.radius

        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        pose = rot @ pose

        # translate
        pose[:3, 3] -= self.look_at

        if self.convention == "opencv":
            pose[:, [1, 2]] *= -1
        elif self.convention == "opengl":
            pass
        else:
            raise ValueError(f"Unknown convention: {self.convention}")
        return pose

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.3 * dx)
        rotvec_y = side * np.radians(-0.3 * dy)
        self.rot = Rotation.from_rotvec(rotvec_x) * Rotation.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        d = np.array([dx, -dy, dz])  # the y axis is flipped
        self.look_at += 2 * self.rot.as_matrix()[:3, :3] @ d * self.radius / self.image_height * math.tan(np.radians(self.fovy) / 2)


def visualize_cameras_to_obj(
    camera_ids,
    camera_params,
    output_path="cameras.obj",
    frustum_scale=0.1,
    axes_scale=0.5,
    target_extrinsic_type="c2w",
):
    """Visualize cameras as frustums in an OBJ file with world axes.

    Args:
        camera_ids: List of camera IDs
        camera_params: Dict mapping camera_id to dict with 'intrinsic' and 'extrinsic'
        output_path: Path to save the OBJ file
        frustum_scale: Scale factor for the camera frustum size
        axes_scale: Scale factor for the world axes
        target_extrinsic_type: Type of extrinsic matrices ('c2w' or 'w2c')
    """
    import os

    # Define camera frustum vertices in camera space
    # Simplified pyramid with 5 vertices (center and 4 corners of frustum)
    frustum_vertices = [
        [0, 0, 0],  # camera center
        [-1, -1, 1],  # bottom-left
        [1, -1, 1],  # bottom-right
        [1, 1, 1],  # top-right
        [-1, 1, 1],  # top-left
    ]

    # Define frustum faces (0-indexed)
    frustum_faces = [
        [0, 1, 2],  # bottom face
        [0, 2, 3],  # right face
        [0, 3, 4],  # top face
        [0, 4, 1],  # left face
        [1, 4, 3, 2],  # front face (quad)
    ]

    # Define world coordinate axes - origin and axis endpoints
    axes_scale = float(axes_scale)
    world_axes = [
        [0, 0, 0],  # origin
        [axes_scale, 0, 0],  # x-axis endpoint (red)
        [0, axes_scale, 0],  # y-axis endpoint (green)
        [0, 0, axes_scale],  # z-axis endpoint (blue)
    ]

    with open(output_path, "w") as f:
        f.write("# Camera visualization OBJ file\n")
        f.write("# Generated by VHAP\n\n")

        # Write material library reference
        mtl_path = os.path.splitext(output_path)[0] + ".mtl"
        f.write(f"mtllib {os.path.basename(mtl_path)}\n\n")

        vertex_count = 0

        # Write world coordinate axes vertices
        f.write("# World coordinate axes\n")
        for v in world_axes:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write X-axis (red)
        f.write("g world_axis_x\n")
        f.write("usemtl red\n")
        f.write(f"l 1 2\n")

        # Write Y-axis (green)
        f.write("g world_axis_y\n")
        f.write("usemtl green\n")
        f.write(f"l 1 3\n")

        # Write Z-axis (blue)
        f.write("g world_axis_z\n")
        f.write("usemtl blue\n")
        f.write(f"l 1 4\n\n")

        vertex_count += len(world_axes)

        # Process each camera
        for idx, cam_id in enumerate(camera_ids):
            f.write(f"# Camera {cam_id}\n")

            # Get camera parameters
            intrinsic = camera_params[cam_id]["intrinsic"]
            extrinsic = camera_params[cam_id]["extrinsic"]

            # Convert from tensor if needed
            if isinstance(intrinsic, torch.Tensor):
                intrinsic = intrinsic.detach().cpu().numpy()
            if isinstance(extrinsic, torch.Tensor):
                extrinsic = extrinsic.detach().cpu().numpy()

            # Convert to camera-to-world if needed
            if target_extrinsic_type == "w2c":
                # Convert to 4x4 matrix if it's 3x4
                if extrinsic.shape[0] == 3 and extrinsic.shape[1] == 4:
                    w2c = np.eye(4)
                    w2c[:3, :] = extrinsic
                else:
                    w2c = extrinsic

                # Create c2w from w2c
                c2w = np.linalg.inv(w2c)
            else:
                # Already c2w, convert to 4x4 if needed
                if extrinsic.shape[0] == 3 and extrinsic.shape[1] == 4:
                    c2w = np.eye(4)
                    c2w[:3, :] = extrinsic
                else:
                    c2w = extrinsic

            # Create scaled frustum vertices
            scaled_frustum = []
            for v in frustum_vertices:
                # Scale the frustum
                scaled_v = [
                    v[0] * frustum_scale,
                    v[1] * frustum_scale,
                    v[2] * frustum_scale,
                ]
                scaled_frustum.append(scaled_v)

            # Transform frustum vertices to world space
            transformed_vertices = []
            for v in scaled_frustum:
                # Convert to homogeneous coordinates
                v_hom = np.array([v[0], v[1], v[2], 1.0])
                # Transform to world space
                v_world = c2w @ v_hom
                transformed_vertices.append(v_world[:3])

            # Write camera frustum vertices
            for v in transformed_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write camera frustum faces
            f.write(f"g camera_{cam_id}\n")
            f.write("usemtl camera\n")
            for face in frustum_faces:
                if len(face) == 3:
                    f.write(
                        f"f {face[0] + vertex_count + 1} {face[1] + vertex_count + 1} {face[2] + vertex_count + 1}\n"
                    )
                elif len(face) == 4:
                    f.write(
                        f"f {face[0] + vertex_count + 1} {face[1] + vertex_count + 1} {face[2] + vertex_count + 1} {face[3] + vertex_count + 1}\n"
                    )

            vertex_count += len(transformed_vertices)
            f.write("\n")

            # Add a small coordinate frame for each camera (showing camera orientation)
            local_axes_scale = frustum_scale * 0.5
            camera_axes = [
                [0, 0, 0],  # camera center
                [local_axes_scale, 0, 0],  # camera x-axis
                [0, local_axes_scale, 0],  # camera y-axis
                [0, 0, local_axes_scale],  # camera z-axis
            ]

            # Transform camera axes to world space
            transformed_axes = []
            for v in camera_axes:
                v_hom = np.array([v[0], v[1], v[2], 1.0])
                v_world = c2w @ v_hom
                transformed_axes.append(v_world[:3])
                f.write(f"v {v_world[0]:.6f} {v_world[1]:.6f} {v_world[2]:.6f}\n")

            # Draw camera coordinate axes
            center_idx = vertex_count + 1
            # X-axis (red)
            f.write(f"g camera_{cam_id}_axis_x\n")
            f.write("usemtl red\n")
            f.write(f"l {center_idx} {center_idx + 1}\n")

            # Y-axis (green)
            f.write(f"g camera_{cam_id}_axis_y\n")
            f.write("usemtl green\n")
            f.write(f"l {center_idx} {center_idx + 2}\n")

            # Z-axis (blue)
            f.write(f"g camera_{cam_id}_axis_z\n")
            f.write("usemtl blue\n")
            f.write(f"l {center_idx} {center_idx + 3}\n\n")

            vertex_count += len(camera_axes)

    # Create material file
    with open(mtl_path, "w") as f:
        f.write("# Material definitions for camera visualization\n\n")

        # Define camera material
        f.write("newmtl camera\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 0.8 0.8 0.8\n")
        f.write("Ks 1.0 1.0 1.0\n")
        f.write("Ns 100.0\n\n")

        # Define axis materials
        f.write("newmtl red\n")
        f.write("Ka 0.1 0.0 0.0\n")
        f.write("Kd 1.0 0.0 0.0\n")
        f.write("Ks 1.0 0.0 0.0\n")
        f.write("Ns 100.0\n\n")

        f.write("newmtl green\n")
        f.write("Ka 0.0 0.1 0.0\n")
        f.write("Kd 0.0 1.0 0.0\n")
        f.write("Ks 0.0 1.0 0.0\n")
        f.write("Ns 100.0\n\n")

        f.write("newmtl blue\n")
        f.write("Ka 0.0 0.0 0.1\n")
        f.write("Kd 0.0 0.0 1.0\n")
        f.write("Ks 0.0 0.0 1.0\n")
        f.write("Ns 100.0\n")

    print(f"Camera visualization saved to {output_path} and {mtl_path}")

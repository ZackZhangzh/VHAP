import os
import sys
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List
import tyro
from rich.console import Console
from rich.logging import RichHandler
import logging

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rename_tool")
console = Console()

@dataclass
class RenameConfig:
    """Configuration for renaming images and camera parameters."""
    
    base_path: Path
    """Base directory containing the dataset (should contain 'images', 'calibration', etc.)"""

    camera_mode: Literal['keep_last_4', 'sequential', 'no_change'] = 'keep_last_4'
    """Mode for renaming camera IDs in camera_params.json:
       - 'keep_last_4': Take the last 4 digits of the ID (e.g., '000001' -> '0001')
       - 'sequential': Rename sequentially from 0 (e.g., '0', '1', '2'...)
       - 'no_change': Do not modify camera IDs
    """

    image_mode: Literal['cam_id_time', 'time_cam_id', 'sequential_cam_0', 'simple_timestep', 'no_change'] = 'sequential_cam_0'
    """Mode for renaming image files:
       - 'cam_id_time': cam_{camera_id}_{timestep}.jpg
       - 'time_cam_id': cam_{timestep}_{camera_id}.jpg
       - 'sequential_cam_0': cam_{index:05d}_0000.jpg (Useful for static camera mode)
       - 'simple_timestep': {index:05d}.jpg (e.g., 00000.jpg)
       - 'no_change': Do not rename images
    """

    dry_run: bool = False
    """If True, only print what would happen without making changes."""

    validate_counts: bool = True
    """If True, check if the number of images matches the number of camera parameters."""


def get_image_files(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
    return sorted([
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ])

def process_camera_params(config: RenameConfig, calibration_dir: Path) -> int:
    json_file_path = calibration_dir / "camera_params.json"
    if not json_file_path.exists():
        logger.warning(f"Camera params file not found: {json_file_path}")
        return 0

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if 'world_2_cam' not in data:
            logger.warning(f"No 'world_2_cam' key in {json_file_path}")
            return 0

        old_world_2_cam = data['world_2_cam']
        new_world_2_cam = {}
        
        # Sort keys to ensure deterministic sequential renaming
        sorted_keys = sorted(old_world_2_cam.keys())
        
        for idx, old_key in enumerate(sorted_keys):
            value = old_world_2_cam[old_key]
            
            if config.camera_mode == 'keep_last_4':
                if len(old_key) >= 4 and old_key.isdigit():
                    new_key = old_key[-4:]
                else:
                    new_key = old_key
            elif config.camera_mode == 'sequential':
                new_key = f"{idx:05d}" # Matches the image renaming style if needed
            else: # no_change
                new_key = old_key

            if new_key != old_key:
                logger.debug(f"Renaming camera ID: {old_key} -> {new_key}")
            
            new_world_2_cam[new_key] = value

        num_cameras = len(new_world_2_cam)
        
        if not config.dry_run:
            data['world_2_cam'] = new_world_2_cam
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Updated {json_file_path} with {num_cameras} cameras.")
        else:
            logger.info(f"[Dry Run] Would update {json_file_path} with {num_cameras} cameras.")

        return num_cameras

    except Exception as e:
        logger.error(f"Error processing {json_file_path}: {e}")
        return 0

def process_images(config: RenameConfig, image_dir: Path):
    files = get_image_files(image_dir)
    num_images = len(files)
    logger.info(f"Found {num_images} images in {image_dir}")

    for idx, file_path in enumerate(files):
        ext = file_path.suffix.lower()
        
        if config.image_mode == 'sequential_cam_0':
            # cam_{timestep:05d}_{camera_id:04d}.jpg where camera_id is 0000
            new_name = f"cam_{idx:05d}_0000{ext}"
        elif config.image_mode == 'cam_id_time':
            # Assuming single camera 0 for now, or need logic to parse original name
            # This is a placeholder for complex logic if needed
            new_name = f"cam_0000_{idx:05d}{ext}"
        elif config.image_mode == 'time_cam_id':
             new_name = f"cam_{idx:05d}_0000{ext}"
        elif config.image_mode == 'simple_timestep':
             new_name = f"{idx:05d}{ext}"
        else:
            continue

        new_path = image_dir / new_name
        
        if new_path != file_path:
            if not config.dry_run:
                file_path.rename(new_path)
                # logger.debug(f"Renamed {file_path.name} -> {new_name}")
            else:
                pass
                # logger.info(f"[Dry Run] Would rename {file_path.name} -> {new_name}")
    
    if not config.dry_run:
        logger.info(f"Renamed {num_images} images in {image_dir}")
    
    return num_images

def main(cfg: RenameConfig):
    if not cfg.base_path.exists():
        logger.error(f"Base path does not exist: {cfg.base_path}")
        return

    logger.info(f"Processing base path: {cfg.base_path}")
    logger.info(f"Modes - Camera: {cfg.camera_mode}, Image: {cfg.image_mode}")

    # 1. Process Calibration
    num_cameras = 0
    calibration_dir = cfg.base_path / "calibration"
    if calibration_dir.exists():
        num_cameras = process_camera_params(cfg, calibration_dir)
    else:
        logger.warning(f"No calibration directory found at {calibration_dir}")

    # 2. Process Images
    image_dirs = [
        d for d in cfg.base_path.iterdir() 
        if d.is_dir() and (d.name == "images" or d.name.startswith("images_") or d.name == "alpha_maps")
    ]

    for img_dir in image_dirs:
        num_images = process_images(cfg, img_dir)
        
        if cfg.validate_counts and num_cameras > 0:
            if num_images != num_cameras:
                logger.warning(f"Mismatch in {img_dir.name}: {num_images} images vs {num_cameras} camera params!")
            else:
                logger.info(f"Validation passed for {img_dir.name}: {num_images} images matches camera count.")

if __name__ == "__main__":
    # Default configuration block - can be modified directly here
    # equivalent to running with command line arguments
    default_cfg = RenameConfig(
        base_path=Path("."), # Replace with your default path
        camera_mode='keep_last_4',
        image_mode='sequential_cam_0',
        dry_run=False,
        validate_counts=True
    )
    
    # If arguments are provided, use tyro to parse them. 
    # Otherwise, use the default configuration defined above.
    if len(sys.argv) > 1:
        tyro.cli(main)
    else:
        logger.info("No arguments provided, using default configuration defined in code.")
        main(default_cfg)

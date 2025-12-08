import shutil
from pathlib import Path
from typing import List
import tyro
import cv2
from tqdm import tqdm


def process_frames_to_images(
    source_frames_dir: Path,
    images_dir: Path,
    target_fps: int = 25,
    source_fps: int = 50,
    crop_to_square: bool = True,
    target_size: int = 2048,
):
    """
    Process sharp frames to create final training images with frame sampling,
    cropping, and resizing.
    """
    print(f"Processing frames for training images")
    print(
        f"  Source FPS: {source_fps} -> Target FPS: {target_fps} (Sub-sampling)")
    print(
        f"  Crop to square: {crop_to_square}, Target Size: {target_size}x{target_size}"
    )

    if images_dir.exists():
        print(
            f"Warning: Output directory {images_dir} already exists. It will be cleared."
        )
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    frame_paths = sorted(
        [
            p
            for p in source_frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
        ]
    )
    if not frame_paths:
        print("Warning: No source frames found!")
        return 0

    # Calculate frame sampling ratio
    frame_skip_ratio = source_fps / target_fps

    processed_count = 0
    print(f"Processing {len(frame_paths)} frames...")
    for i, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
        # Sample frames based on target FPS
        if i % max(1, round(frame_skip_ratio)) != 0:
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        H, W = image.shape[:2]

        if crop_to_square:
            # Crop to center square
            if W != H:
                short_dim = min(W, H)
                crop_left = (W - short_dim) // 2
                crop_top = (H - short_dim) // 2
                image = image[
                    crop_top: crop_top + short_dim, crop_left: crop_left + short_dim
                ]

            # Resize to target size
            if image.shape[0] != target_size:
                image = cv2.resize(
                    image, (target_size, target_size), interpolation=cv2.INTER_AREA
                )

        # Save processed image with new sequential name
        output_path = images_dir / f"{processed_count:05d}.jpg"
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_count += 1

    print(f"[Frame Processing] Generated {processed_count} training images")
    return processed_count


def create_downsampled_images(
    images_dir: Path, output_dir: Path, downsample_scales: List[int]
):
    """
    Create downsampled versions of images for multi-resolution training.
    """
    if not downsample_scales:
        return

    print(f"Creating downsampled images with scales: {downsample_scales}")

    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
        ]
    )
    if not image_paths:
        print("Warning: No images found for downsampling!")
        return

    for scale in downsample_scales:
        if scale <= 1:
            continue
        print(f"  Creating images_{scale}...")
        scaled_dir = output_dir / f"images_{scale}"
        if scaled_dir.exists():
            shutil.rmtree(scaled_dir)
        scaled_dir.mkdir(exist_ok=True)

        for image_path in tqdm(image_paths, desc=f"Downsampling x{scale}"):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            H, W = image.shape[:2]
            new_H, new_W = H // scale, W // scale

            resized_image = cv2.resize(
                image, (new_W, new_H), interpolation=cv2.INTER_AREA
            )

            output_path = scaled_dir / image_path.name
            cv2.imwrite(str(output_path), resized_image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])


def main(
    sequence_dir: Path,
    input_frames_folder: str = "frames_sharp",
    target_fps: int = 25,
    source_fps: int = 50,
    downsample_scales: List[int] = [2, 4, 8],
    crop_to_square: bool = True,
    target_size: int = 2048,
):
    """
    Part 2: Processes frames to create final training images.
    This includes frame sub-sampling, cropping, resizing, and creating
    downsampled versions for multi-resolution training.

    Args:
        sequence_dir: The base output directory for the sequence (e.g., data/person_01).
        input_frames_folder: The name of the folder inside sequence_dir containing the frames to process.
        target_fps: Target frames per second for the final training images.
        source_fps: The original frame rate of the input frames (from Part 1).
        downsample_scales: List of scales to downsample the images by (e.g., 2, 4, 8).
        crop_to_square: Whether to crop images to a square format.
        target_size: The target resolution for the final high-res images.
    """
    source_frames_dir = sequence_dir / input_frames_folder
    if not source_frames_dir.is_dir():
        raise FileNotFoundError(
            f"Input frames directory not found: {source_frames_dir}"
        )

    print(f"=== NeRSemble Video Preprocessing - Part 2: Frame Processing ===")
    print(f"Input frames: {source_frames_dir}")
    print(f"Output directory: {sequence_dir}")

    # --- Step 1: Process frames to training images ---
    print(f"\n=== Step 1: Processing Frames to Training Images ===")
    images_dir = sequence_dir / "images"
    num_training_images = process_frames_to_images(
        source_frames_dir,
        images_dir,
        target_fps=target_fps,
        source_fps=source_fps,
        crop_to_square=crop_to_square,
        target_size=target_size,
    )

    # --- Step 2: Create downsampled versions ---
    if downsample_scales:
        print(f"\n=== Step 2: Creating Downsampled Images ===")
        create_downsampled_images(images_dir, sequence_dir, downsample_scales)
    else:
        print(f"\n=== Step 2: Skipping Downsampling ===")

    # --- Summary ---
    print(f"\n=== Part 2 Summary ===")
    print(f"Final training images generated: {num_training_images}")
    print(
        f"Downsampled versions created: {len([s for s in downsample_scales if s > 1])} scales"
    )
    print(f"\nHigh-resolution images are in: {images_dir}")
    for scale in downsample_scales:
        if scale > 1:
            print(
                f"Downsampled x{scale} images are in: {sequence_dir / f'images_{scale}'}"
            )

    print(f"\n=== Part 2 Complete ===")


if __name__ == "__main__":
    tyro.cli(main)

from pathlib import Path
from tqdm import tqdm
from typing import Literal, Optional, List
import tyro
from PIL import Image, ImageOps
import torch
from vhap.data.image_folder_dataset import ImageFolderDataset
from torch.utils.data import DataLoader
from BackgroundMattingV2.model import MattingRefine
from BackgroundMattingV2.asset import get_weights_path
import shutil


def process_and_copy_images(
    source_path: Path,
    sequence_path: Path,
    downsample_scales: List[int],
    crop_to_square: bool = True,
    target_size: Optional[int] = None,
    output_name_format: str = "{i:06d}.jpg"
):
    """
    Copies images from source_path to sequence_path/frames, renames them,
    and creates sequentially downsampled versions in sequence_path/images_*.
    """
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    # Step 1: Copy and rename original images to the 'frames' directory
    frames_dir = sequence_path / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying and renaming images from {source_path} to {frames_dir}")
    image_paths = sorted(list(source_path.glob('*.jpg')) +
                         list(source_path.glob('*.jpeg')) +
                         list(source_path.glob('*.png')) +
                         list(source_path.glob('*.JPG')) +
                         list(source_path.glob('*.JPEG')) +
                         list(source_path.glob('*.PNG')))
    if not image_paths:
        raise FileNotFoundError(
            f"No .jpg, .jpeg, .png, .JPG, .JPEG, or .PNG images found in {source_path}")

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        # Use the output_name_format to generate the new filename
        new_name = output_name_format.format(i=i)
        shutil.copy2(image_path, frames_dir / new_name)

    print(f"Finished copying {len(image_paths)} images.")

    # Step 2: Create full-resolution 'images' directory from 'frames'
    images_dir = sequence_path / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating full-resolution images in {images_dir}")

    source_image_paths = sorted(list(frames_dir.glob('*.*')))
    for image_path in tqdm(source_image_paths, total=len(source_image_paths)):
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if crop_to_square:
            w, h = img.size
            if w != h:
                short_dim = min(w, h)
                crop_left = (w - short_dim) // 2
                crop_top = (h - short_dim) // 2
                crop_right = crop_left + short_dim
                crop_bottom = crop_top + short_dim
                img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        if target_size is not None and img.size[0] != target_size:
            img = img.resize((target_size, target_size),
                             Image.Resampling.LANCZOS)

        new_path = images_dir / (image_path.stem + '.jpg')
        img.convert('RGB').save(new_path, quality=95)

    # Step 3: Sequentially downsample images
    # Start from the full-resolution 'images' dir
    previous_dir = images_dir
    if downsample_scales:
        # Sort scales to ensure we downsample progressively (e.g., 2, then 4, then 8)
        sorted_scales = sorted(downsample_scales)
        previous_scale = 1

        for scale in sorted_scales:
            if scale % previous_scale != 0:
                raise ValueError(
                    f"Downsample scales must be multiples of each other. Cannot get from {previous_scale}x to {scale}x.")

            downsample_factor = scale // previous_scale
            if downsample_factor <= 1:
                continue

            current_dir = sequence_path / f'images_{scale}'
            current_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Creating 1/{scale} resolution images from 1/{previous_scale} resolution images")

            source_images = sorted(list(previous_dir.glob('*.jpg')))
            for image_path in tqdm(source_images, total=len(source_images)):
                img = Image.open(image_path)
                w, h = img.size
                new_w, new_h = w // downsample_factor, h // downsample_factor
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                new_path = current_dir / image_path.name
                img.save(new_path, quality=95)

            previous_dir = current_dir
            previous_scale = scale


def robust_video_matting(image_dir: Path, N_warmup: Optional[int] = 10):
    print(f'Running robust video matting on images in {image_dir}')
    # model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda()
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda()

    dataset = ImageFolderDataset(image_folder=image_dir)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1)

    rec = [None] * 4  # Initial recurrent states.
    downsample_ratio = 0.5  # (for videos in 512x512)
    for item in tqdm(dataloader):
        rgb = item['rgb']
        rgb = rgb.permute(0, 3, 1, 2).float().cuda() / 255
        with torch.no_grad():
            if N_warmup and N_warmup > 0:
                # use the first frame to warm up the recurrent states
                temp_rec = rec
                for _ in range(N_warmup):
                    _, _, *temp_rec = model(rgb, *temp_rec, downsample_ratio)
                N_warmup = 0  # only warmup once

            fgr, pha, *rec = model(rgb, *rec, downsample_ratio)

        alpha = (pha[0, 0] * 255).cpu().numpy()
        alpha = Image.fromarray(alpha.astype('uint8'))
        alpha_path = Path(
            str(item['image_path'][0]).replace('images', 'alpha_maps'))
        if not alpha_path.parent.exists():
            alpha_path.parent.mkdir(parents=True)
        alpha.save(alpha_path)


def background_matting_v2(
    image_dir: Path,
    background_folder: Path = Path('../../BACKGROUND'),
    model_backbone: Literal['resnet101',
                            'resnet50', 'mobilenetv2'] = 'resnet101',
    model_backbone_scale: float = 0.25,
    model_refine_mode: Literal['full', 'sampling',
                               'thresholding'] = 'thresholding',
    model_refine_sample_pixels: int = 80_000,
    model_refine_threshold: float = 0.01,
    model_refine_kernel_size: int = 3,
):
    model = MattingRefine(
        model_backbone,
        model_backbone_scale,
        model_refine_mode,
        model_refine_sample_pixels,
        model_refine_threshold,
        model_refine_kernel_size
    )

    weights_path = get_weights_path(model_backbone)

    model = model.cuda().eval()
    model.load_state_dict(torch.load(
        weights_path, map_location='cuda', weights_only=True))

    dataset = ImageFolderDataset(
        image_folder=image_dir,
        background_folder=background_folder,
        background_fname2camId=lambda x: x.split('.')[0].split('_')[1],
        image_fname2camId=lambda x: x.split('.')[0].split('_')[1],
    )
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1)

    for item in tqdm(dataloader):
        src = item['rgb']
        bgr = item['background']
        src = src.permute(0, 3, 1, 2).float().cuda() / 255
        bgr = bgr.permute(0, 3, 1, 2).float().cuda() / 255

        with torch.no_grad():
            pha, fgr, _, _, err, ref = model(src, bgr)

        alpha = (pha[0, 0] * 255).cpu().numpy()
        alpha = Image.fromarray(alpha.astype('uint8'))
        alpha_path = Path(
            str(item['image_path'][0]).replace('images', 'alpha_maps'))
        if not alpha_path.parent.exists():
            alpha_path.parent.mkdir(parents=True)
        alpha.save(alpha_path)


def main(
    source_path: Path,
    sequence_path: Path,
    downsample_scales: List[int] = [],
    matting_method: Optional[Literal['robust_video_matting',
                                     'background_matting_v2']] = None,
    background_folder: Path = Path('../../BACKGROUND'),
    crop_to_square: bool = True,
    target_size: Optional[int] = None,
    output_name_format: str = "{i:06d}.jpg",
):
    """
    Preprocesses a sequence of images by renaming, copying, and downsampling them.
    Optionally, performs foreground matting.

    :param source_path: Path to the directory containing the source images.
    :param sequence_path: Path to the output directory for the processed sequence.
    :param downsample_scales: List of integer scales to downsample the images (e.g., [2, 4, 8]).
    :param matting_method: The foreground matting method to use.
    :param background_folder: Path to the background images for background_matting_v2.
    :param crop_to_square: Whether to crop images to a square format.
    :param target_size: The target resolution for the final high-res images.
    :param output_name_format: Format string for output filenames. Use {i} for image index (e.g., "cam_{i:06d}_000000.jpg").
    """

    # Process images: copy, rename, and create downsampled versions
    process_and_copy_images(
        source_path,
        sequence_path,
        downsample_scales,
        crop_to_square=crop_to_square,
        target_size=target_size,
        output_name_format=output_name_format
    )

    # Foreground matting on the full-resolution images
    if matting_method:
        image_dir = sequence_path / 'images'
        if not image_dir.exists():
            print(
                f"Warning: Image directory for matting not found at {image_dir}. Skipping matting.")
            return

        print(f"\nPerforming foreground matting with method: {matting_method}")
        if matting_method == 'robust_video_matting':
            robust_video_matting(image_dir)
        elif matting_method == 'background_matting_v2':
            background_matting_v2(
                image_dir, background_folder=background_folder)
        else:
            # This case should be prevented by tyro's Literal, but included for safety
            raise ValueError(f'Unknown matting method: {matting_method}')

    print("\nProcessing finished.")


if __name__ == '__main__':
    tyro.cli(main)

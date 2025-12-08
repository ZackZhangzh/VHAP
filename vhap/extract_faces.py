from tqdm import tqdm  # 在文件头部导入
import argparse
import os
from pathlib import Path
from PIL import Image


def get_image_info(folder):
    info = {}
    for f in Path(folder).iterdir():
        if f.is_file() and not f.name.startswith('.'):
            info[f.stem] = f
    return info


def apply_masks(images_path, masks_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    img_info = get_image_info(images_path)
    mask_info = get_image_info(masks_path)

    common = img_info.keys() & mask_info.keys()
    if not common:
        print("没有找到匹配的文件名。")
        return

    for key in tqdm(sorted(common), desc="Extracting faces", unit="img"):
        img_path = img_info[key]
        mask_path = mask_info[key]

        with Image.open(img_path).convert("RGBA") as img, \
                Image.open(mask_path).convert("L") as mask:
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.LANCZOS)
            img.putalpha(mask)
            out_path = Path(output_path) / f"{key}.png"
            img.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 mask 提取人脸区域")
    parser.add_argument("--images_path", required=True, help="原图目录")
    parser.add_argument("--masks_path", required=True, help="掩码目录")
    parser.add_argument("--output_path", required=True, help="输出目录")
    args = parser.parse_args()

    apply_masks(args.images_path, args.masks_path, args.output_path)

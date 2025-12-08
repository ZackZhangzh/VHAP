import os
import sys
import json


def rename_camera_ids_in_json(json_file_path):
    """Rename camera IDs from 6-digit format (000000) to 4-digit format (0000)"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if 'world_2_cam' in data:
            old_world_2_cam = data['world_2_cam']
            new_world_2_cam = {}
            for old_key, value in old_world_2_cam.items():
                # Convert 6-digit format to 4-digit format
                if len(old_key) == 6 and old_key.isdigit():
                    new_key = old_key[-4:]  # Take last 4 digits
                    new_world_2_cam[new_key] = value
                    print(f"  Renamed camera ID: {old_key} -> {new_key}")
                else:
                    # Keep original key if it doesn't match expected format
                    new_world_2_cam[old_key] = value

            data['world_2_cam'] = new_world_2_cam

            # Write back to file
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Successfully updated camera IDs in {json_file_path}")
            return True

    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return False


def rename_images_in_dir(directory):
    # 支持的图片扩展名
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
    # 获取所有图片文件
    files = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in exts
    ]
    # 按文件名排序
    files.sort()
    # 重命名
    for idx, filename in enumerate(files):
        ext = os.path.splitext(filename)[1].lower()
        # new_name = f"cam_0000_{idx:04d}.jpg"
        new_name = f"cam_{idx:05d}_0000.jpg"
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        os.rename(src, dst)
        print(f"{filename} -> {new_name}")


def process_directories(base_path):
    """Process all alpha_maps, images, images_*, and calibration directories under base_path"""
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == "alpha_maps" or dir_name == "images" or dir_name.startswith("images_"):
                target_dir = os.path.join(root, dir_name)
                print(f"\nProcessing image directory: {target_dir}")
                rename_images_in_dir(target_dir)
            elif dir_name == "calibration":
                calibration_dir = os.path.join(root, dir_name)
                camera_params_file = os.path.join(
                    calibration_dir, "camera_params.json")
                if os.path.exists(camera_params_file):
                    print(
                        f"\nProcessing calibration file: {camera_params_file}")
                    rename_camera_ids_in_json(camera_params_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "1"

    print(f"Processing base path: {base_path}")
    process_directories(base_path)

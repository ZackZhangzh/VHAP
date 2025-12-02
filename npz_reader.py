#!/usr/bin/env python3
"""
npz_reader.py
快速查看 .npz 文件内容：列出所有键、数组形状、数据类型、部分数值。
用法：
    python npz_reader.py path/to/file.npz
"""

import sys
from textwrap import indent
import numpy as np
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("用法: python npz_reader.py <npz_file>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"加载失败: {e}")
        sys.exit(1)

    print("=" * 60)
    print(f"NPZ 文件: {path}")
    print("=" * 60)
    print(f"包含 {len(data.files)} 个键:\n")

    for key in data.files:
        # print("-" * 60)
        arr = data[key]
        print(f"--- 键名: {key} ---")
        print(f"  类型: {type(arr)}")
        if isinstance(arr, np.ndarray):
            print(f"  数组形状: {arr.shape}")
            print(f"  数据类型: {arr.dtype}")
            # 打印前若干项
            flat = arr.ravel()
            preview = flat[: min(10, flat.size)]
            # 紧凑预览
            preview = np.array2string(preview, separator=", ", max_line_width=80)
            print(f"  前 {len(flat[: min(10, flat.size)])} 项: {preview}")
        else:
            print(f"  值: {arr}")
        print()

    data.close()


if __name__ == "__main__":
    main()

"""
设置数据目录链接

将 bq_corpus 和 lcqmc 数据集链接到当前项目的 data 目录
"""

import os
import sys
from pathlib import Path

# 源数据目录（week08 的数据）
SOURCE_DATA = Path(__file__).parent.parent / "week08" / "data"

# 目标数据目录（当前项目）
TARGET_DATA = Path(__file__).parent / "data"

def setup_data_links():
    """创建数据目录链接"""
    SOURCE_DATA = Path(__file__).parent.parent / "week08" / "data"
    TARGET_DATA = Path(__file__).parent / "data"

    if not SOURCE_DATA.exists():
        print(f"源数据目录不存在: {SOURCE_DATA}")
        return False

    TARGET_DATA.mkdir(parents=True, exist_ok=True)

    for dataset in ["bq_corpus", "lcqmc"]:
        src_dir = SOURCE_DATA / dataset
        dst_dir = TARGET_DATA / dataset

        if dst_dir.exists():
            print(f"  {dataset}: 已存在，跳过")
            continue

        if os.name == "nt":
            # Windows: 使用 junction point
            try:
                os.symlink(str(dst_dir), str(src_dir), target_is_directory=True)
                print(f"  {dataset}: 已创建符号链接 -> {src_dir}")
            except OSError:
                # 如果符号链接失败，复制目录
                import shutil
                shutil.copytree(src_dir, dst_dir)
                print(f"  {dataset}: 已复制到 {dst_dir}")
        else:
            # Unix: 使用符号链接
            dst_dir.symlink_to(src_dir)
            print(f"  {dataset}: 已创建符号链接 -> {src_dir}")

    print("\n数据设置完成!")
    return True


if __name__ == "__main__":
    setup_data_links()

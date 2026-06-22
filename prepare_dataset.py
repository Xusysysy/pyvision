"""
数据集准备工具 — prepare_dataset.py

将采集的图片整理为 YOLO 训练格式:
  1. 将智能/普通眼镜图片 + 标注文件划分训练集/验证集
  2. 负样本 (空桌面) 加入训练集 (无边界框，作为背景学习)

用法:
  python prepare_dataset.py --val-ratio 0.2
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="准备 YOLO 训练数据集")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    train_img = Path("dataset/images/train")
    val_img = Path("dataset/images/val")
    train_lbl = Path("dataset/labels/train")
    val_lbl = Path("dataset/labels/val")

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    base = Path("dataset/images")

    def find_pairs(image_dir):
        img_dir = base / image_dir
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        pairs = []
        unlabeled = []
        for img_path in images:
            lbl_path = base.parent / "labels" / image_dir / img_path.name.replace(img_path.suffix, ".txt")
            alt_lbl = img_dir.parent.parent / "labels" / img_path.name.replace(img_path.suffix, ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
            elif alt_lbl.exists():
                pairs.append((img_path, alt_lbl))
            elif len(base.parent / "labels" / image_dir) > 0:
                # 目录存在但标签不匹配
                pass
            else:
                unlabeled.append(img_path)
        return images, pairs, unlabeled

    # 智能眼镜
    smart_images, smart_pairs, smart_unlabeled = find_pairs("smart")
    print(f"智能眼镜: {len(smart_images)} 张, 已标注 {len(smart_pairs)} 张")
    if smart_unlabeled:
        print(f"  未标注 {len(smart_unlabeled)} 张 — 请用 labelImg 标注")

    # 普通眼镜
    regular_images, regular_pairs, regular_unlabeled = find_pairs("regular")
    print(f"普通眼镜: {len(regular_images)} 张, 已标注 {len(regular_pairs)} 张")
    if regular_unlabeled:
        print(f"  未标注 {len(regular_unlabeled)} 张 — 请用 labelImg 标注")

    # 合并已标注对
    all_pairs = smart_pairs + regular_pairs
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * (1 - args.val_ratio))
    train_pairs = all_pairs[:split]
    val_pairs = all_pairs[split:]

    for img_path, lbl_path in train_pairs:
        shutil.copy2(img_path, train_img / img_path.name)
        shutil.copy2(lbl_path, train_lbl / lbl_path.name)

    for img_path, lbl_path in val_pairs:
        shutil.copy2(img_path, val_img / img_path.name)
        shutil.copy2(lbl_path, val_lbl / lbl_path.name)

    # 统计每类数量
    train_smart = sum(1 for p in train_pairs if "smart" in str(p[0]))
    train_regular = sum(1 for p in train_pairs if "regular" in str(p[0]))
    val_smart = sum(1 for p in val_pairs if "smart" in str(p[0]))
    val_regular = sum(1 for p in val_pairs if "regular" in str(p[0]))

    print(f"\n训练集: {len(train_pairs)} 张 (智能: {train_smart}, 普通: {train_regular})")
    print(f"验证集: {len(val_pairs)} 张 (智能: {val_smart}, 普通: {val_regular})")

    # 负样本 (空桌面)
    neg_dir = base / "negative"
    neg_images = sorted(list(neg_dir.glob("*.jpg")) + list(neg_dir.glob("*.png")))
    if neg_images:
        random.shuffle(neg_images)
        neg_split = int(len(neg_images) * (1 - args.val_ratio))
        for img_path in neg_images[:neg_split]:
            shutil.copy2(img_path, train_img / img_path.name)
        for img_path in neg_images[neg_split:]:
            shutil.copy2(img_path, val_img / img_path.name)
        print(f"负样本: 训练 {neg_split} 张, 验证 {len(neg_images) - neg_split} 张")

    print("\n准备完成。下一步:")
    print("  python train.py --epochs 100 --imgsz 320")


if __name__ == "__main__":
    main()

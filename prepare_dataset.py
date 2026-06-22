"""
数据集准备工具 — prepare_dataset.py

将采集的正/负样本图片整理为 YOLO 训练格式:
  1. 将正样本图片 + 标注文件移动到 dataset/images/train/ 和 dataset/labels/train/
  2. 按比例分割训练集/验证集
  3. 可选: 将负样本加入训练集 (无边界框，作为背景训练)

用法:
  python prepare_dataset.py --positive dataset/images/positive --negative dataset/images/negative --val-ratio 0.2
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="准备 YOLO 训练数据集")
    parser.add_argument("--positive", type=str, default="dataset/images/positive",
                        help="正样本图片目录 (佩戴智能眼镜)")
    parser.add_argument("--negative", type=str, default="dataset/images/negative",
                        help="负样本图片目录 (未佩戴)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--include-negative", action="store_true", default=True,
                        help="是否包含负样本 (无标签图片)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    train_img = Path("dataset/images/train")
    val_img = Path("dataset/images/val")
    train_lbl = Path("dataset/labels/train")
    val_lbl = Path("dataset/labels/val")

    train_img.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    pos_dir = Path(args.positive)
    neg_dir = Path(args.negative)

    # 获取所有正样本图片
    pos_images = sorted(list(pos_dir.glob("*.jpg")) + list(pos_dir.glob("*.png")))
    if not pos_images:
        print(f"警告: 未在 {pos_dir} 中找到正样本图片")

    # 匹配标注文件 (labelImg 生成的 .txt 与图片同名)
    paired = []
    unlabeled = []
    for img_path in pos_images:
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        # 也尝试在同目录查找
        alt_label = pos_dir.parent.parent / "labels" / img_path.name.replace(img_path.suffix, ".txt")
        if label_path.exists():
            paired.append((img_path, label_path))
        elif alt_label.exists():
            paired.append((img_path, alt_label))
        else:
            unlabeled.append(img_path)

    print(f"正样本图片: {len(pos_images)} 张")
    print(f"  已标注:   {len(paired)} 张")
    print(f"  未标注:   {len(unlabeled)} 张")

    # 分割
    random.shuffle(paired)
    split = int(len(paired) * (1 - args.val_ratio))
    train_pairs = paired[:split]
    val_pairs = paired[split:]

    # 复制已标注图片
    for img_path, lbl_path in train_pairs:
        dst_img = train_img / img_path.name
        dst_lbl = train_lbl / lbl_path.name
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)

    for img_path, lbl_path in val_pairs:
        dst_img = val_img / img_path.name
        dst_lbl = val_lbl / lbl_path.name
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)

    print(f"训练集: {len(train_pairs)} 张 (已标注)")
    print(f"验证集: {len(val_pairs)} 张 (已标注)")

    # 负样本 (无标签)
    if args.include_negative:
        neg_images = sorted(list(neg_dir.glob("*.jpg")) + list(neg_dir.glob("*.png")))
        if neg_images:
            random.shuffle(neg_images)
            neg_split = int(len(neg_images) * (1 - args.val_ratio))
            neg_train = neg_images[:neg_split]
            neg_val = neg_images[neg_split:]

            for img_path in neg_train:
                dst_img = train_img / img_path.name
                shutil.copy2(img_path, dst_img)

            for img_path in neg_val:
                dst_img = val_img / img_path.name
                shutil.copy2(img_path, dst_img)

            print(f"负样本: 训练集 {len(neg_train)} 张, 验证集 {len(neg_val)} 张")
        else:
            print(f"警告: 未在 {neg_dir} 中找到负样本图片")

    # 未标注的正样本: 放入训练集备用手动标注
    if unlabeled:
        print(f"\n未标注的正样本 ({len(unlabeled)} 张) 跳过。请用 labelImg 标注后重新运行。")

    print("\n数据集准备完成。标注提示:")
    print(f"  pip install labelImg")
    print(f"  labelImg {args.positive}")
    print(f"  (选择 YOLO 格式保存, 类别名 smart_glasses)")


if __name__ == "__main__":
    main()

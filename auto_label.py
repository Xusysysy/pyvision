"""
自动标注工具 — auto_label.py

原理: 摄像头固定俯拍桌面 → 拍一张空桌面作为背景 →
     对每张眼镜图片做背景减除 → 提取轮廓 → 自动生成 YOLO 标注。
     默认类别: smart/ 目录 → class 0, regular/ 目录 → class 1

用法:
  # 1. 先采集空桌面 (N 键一张即可，作为背景)
  # 2. 再采集智能/普通眼镜图片 (P/R 键)
  # 3. 运行自动标注
  python auto_label.py

  # 预览模式: 标注前先逐个预览，按 Y 接受 / N 跳过 / E 手动编辑
  python auto_label.py --preview
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def find_glasses_bbox(background, image, min_area=500):
    """通过背景减除找到眼镜区域，返回 (x, y, w, h) 或 None"""
    # 转为灰度
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 背景减除
    diff = cv2.absdiff(bg_gray, img_gray)

    # 自适应阈值
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 形态学操作：去噪 + 填充空洞
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, thresh

    # 取面积最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)

    if area < min_area:
        return None, thresh

    x, y, w, h = cv2.boundingRect(max_contour)

    # 稍微扩展框
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + pad * 2)
    h = min(image.shape[0] - y, h + pad * 2)

    return (x, y, w, h), thresh


def yolo_format(bbox, img_w, img_h):
    """将像素坐标转换为 YOLO 归一化格式"""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def auto_label_directory(image_dir, label_dir, class_id, background, args):
    """对目录下所有图片自动标注"""
    img_paths = sorted(list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png")))
    if not img_paths:
        print(f"  目录为空: {image_dir}")
        return 0, 0

    Path(label_dir).mkdir(parents=True, exist_ok=True)
    auto_count = 0
    skip_count = 0

    for img_path in img_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        if image.shape != background.shape:
            # resize 到一致
            image = cv2.resize(image, (background.shape[1], background.shape[0]))

        result = find_glasses_bbox(background, image)
        if result is None:
            skip_count += 1
            continue

        bbox, thresh = result

        label_file = Path(label_dir) / img_path.name.replace(img_path.suffix, ".txt")

        if args.preview:
            display = image.copy()
            x, y, w, h = bbox
            color = (0, 255, 0) if class_id == 0 else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            label = "smart_glasses" if class_id == 0 else "regular_glasses"
            cv2.putText(display, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Preview - Y=Accept N=Skip E=Edit Later", display)
            cv2.imshow("Threshold", thresh)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('y'):
                pass  # 接受
            elif key == ord('e'):
                auto_count += 1
                with open(label_file, 'w') as f:
                    f.write(f"{class_id} {yolo_format(bbox, image.shape[1], image.shape[0])}\n")
                continue  # 留标注但标记待编辑
            else:
                continue  # 跳过
        else:
            pass  # 非预览模式直接写

        auto_count += 1
        with open(label_file, 'w') as f:
            f.write(f"{class_id} {yolo_format(bbox, image.shape[1], image.shape[0])}\n")

    if args.preview:
        cv2.destroyAllWindows()

    return auto_count, skip_count


def main():
    parser = argparse.ArgumentParser(description="自动标注眼镜数据集")
    parser.add_argument("--background", type=str, default=None,
                        help="背景图路径 (空桌面)，默认取第一张 negative 图片")
    parser.add_argument("--preview", action="store_true", default=True,
                        help="预览标注结果，逐张确认")
    parser.add_argument("--no-preview", action="store_true",
                        help="跳过预览，直接全部自动标注")
    parser.add_argument("--min-area", type=int, default=500,
                        help="最小轮廓面积阈值 (像素)")
    args = parser.parse_args()

    if args.no_preview:
        args.preview = False

    base = Path("dataset/images")

    # 加载背景图
    if args.background:
        background = cv2.imread(args.background)
    else:
        neg_dir = base / "negative"
        neg_images = sorted(list(neg_dir.glob("*.jpg")) + list(neg_dir.glob("*.png")))
        if not neg_images:
            print("错误: 请先采集一张空桌面图片 (按 N 键).")
            print("      或者手动指定: python auto_label.py --background <path>")
            return
        background = cv2.imread(str(neg_images[0]))
        print(f"背景图: {neg_images[0]}")

    if background is None:
        print("错误: 无法读取背景图")
        return

    bg_h, bg_w = background.shape[:2]
    print(f"背景分辨率: {bg_w}x{bg_h}")
    print(f"最小区域阈值: {args.min_area} px")
    print()

    if args.preview:
        print("预览模式: Y=接受标注  N=跳过  E=保留但留待编辑")
        print()

    # 处理智能眼镜
    print("=== 智能眼镜 (class 0) ===")
    auto, skip = auto_label_directory(
        base / "smart", base.parent / "labels" / "smart", 0, background, args)
    print(f"  自动标注: {auto} 张, 跳过: {skip} 张")

    # 处理普通眼镜
    print("=== 普通眼镜 (class 1) ===")
    auto, skip = auto_label_directory(
        base / "regular", base.parent / "labels" / "regular", 1, background, args)
    print(f"  自动标注: {auto} 张, 跳过: {skip} 张")

    print("\n完成。下一步:")
    print("  labelImg dataset/images/smart    # 检查修正标注")
    print("  labelImg dataset/images/regular  # 检查修正标注")
    print("  python prepare_dataset.py")
    print("  python train.py")


if __name__ == "__main__":
    main()

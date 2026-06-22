"""
数据采集工具 — collect_data.py
用法: python collect_data.py [--camera 0] [--width 640] [--height 480]

按键:
  P / Space  → 保存为正样本 (佩戴智能眼镜) → dataset/images/positive/
  N          → 保存为负样本 (未佩戴智能眼镜) → dataset/images/negative/
  Q / Esc    → 退出
"""

import cv2
import os
import argparse
from datetime import datetime
from pathlib import Path


def ensure_dirs(base: str):
    pos = os.path.join(base, "dataset", "images", "positive")
    neg = os.path.join(base, "dataset", "images", "negative")
    os.makedirs(pos, exist_ok=True)
    os.makedirs(neg, exist_ok=True)
    return pos, neg


def main():
    parser = argparse.ArgumentParser(description="智能眼镜数据集采集工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--width", type=int, default=640, help="采集分辨率宽度")
    parser.add_argument("--height", type=int, default=480, help="采集分辨率高度")
    parser.add_argument("--output", type=str, default=".", help="输出根目录")
    args = parser.parse_args()

    pos_dir, neg_dir = ensure_dirs(args.output)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    pos_count = len(list(Path(pos_dir).glob("*.jpg")))
    neg_count = len(list(Path(neg_dir).glob("*.jpg")))

    print(f"摄像头已打开: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"正样本 (智能眼镜): {pos_count} 张 → {pos_dir}")
    print(f"负样本 (无眼镜):   {neg_count} 张 → {neg_dir}")
    print()
    print("按键说明:")
    print("  P / Space  → 保存正样本 (佩戴智能眼镜)")
    print("  N          → 保存负样本 (未佩戴)")
    print("  Q / Esc    → 退出")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("采集帧失败")
            break

        display = frame.copy()

        cv2.putText(display, f"POS: {pos_count}  NEG: {neg_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "P=Positive  N=Negative  Q=Quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 65), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cv2.putText(display, f"POS: {pos_count}  NEG: {neg_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "P/Space=Positive  N=Negative  Q/Esc=Quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Data Collection - Smart Glasses", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('p') or key == ord(' '):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(pos_dir, f"pos_{ts}.jpg")
            cv2.imwrite(filename, frame)
            pos_count += 1
            print(f"[POS #{pos_count}] 已保存: {filename}")
        elif key == ord('n'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(neg_dir, f"neg_{ts}.jpg")
            cv2.imwrite(filename, frame)
            neg_count += 1
            print(f"[NEG #{neg_count}] 已保存: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"采集结束。正样本: {pos_count} 张, 负样本: {neg_count} 张")


if __name__ == "__main__":
    main()

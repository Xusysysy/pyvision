"""
数据采集工具 — collect_data.py
用法: python collect_data.py [--camera 0] [--width 640] [--height 480]

将眼镜平方桌面，USB 摄像头俯拍采集。

按键:
  P / Space  → 保存智能眼镜 (有摄像头/电池的) → dataset/images/smart/
  R          → 保存普通眼镜 (仅镜框镜片)      → dataset/images/regular/
  N          → 保存空桌面 (负样本/背景)       → dataset/images/negative/
  Q / Esc    → 退出
"""

import cv2
import os
import argparse
from datetime import datetime
from pathlib import Path


def ensure_dirs(base: str):
    smart = os.path.join(base, "dataset", "images", "smart")
    regular = os.path.join(base, "dataset", "images", "regular")
    neg = os.path.join(base, "dataset", "images", "negative")
    os.makedirs(smart, exist_ok=True)
    os.makedirs(regular, exist_ok=True)
    os.makedirs(neg, exist_ok=True)
    return smart, regular, neg


def main():
    parser = argparse.ArgumentParser(description="智能眼镜数据集采集工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--width", type=int, default=640, help="采集分辨率宽度")
    parser.add_argument("--height", type=int, default=480, help="采集分辨率高度")
    parser.add_argument("--output", type=str, default=".", help="输出根目录")
    args = parser.parse_args()

    smart_dir, regular_dir, neg_dir = ensure_dirs(args.output)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    smart_count = len(list(Path(smart_dir).glob("*.jpg")))
    regular_count = len(list(Path(regular_dir).glob("*.jpg")))
    neg_count = len(list(Path(neg_dir).glob("*.jpg")))

    print(f"摄像头已打开: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"智能眼镜: {smart_count} 张 → {smart_dir}")
    print(f"普通眼镜: {regular_count} 张 → {regular_dir}")
    print(f"空桌面:   {neg_count} 张 → {neg_dir}")
    print()
    print("按键说明:")
    print("  P / Space  → 保存智能眼镜")
    print("  R          → 保存普通眼镜")
    print("  N          → 保存空桌面")
    print("  Q / Esc    → 退出")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("采集帧失败")
            break

        display = frame.copy()

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 65), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cv2.putText(display, f"Smart: {smart_count}  Regular: {regular_count}  Neg: {neg_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(display, "P/Space=智能眼镜  R=普通眼镜  N=空桌面  Q/Esc=退出", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Data Collection - Smart Glasses", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('p') or key == ord(' '):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(smart_dir, f"smart_{ts}.jpg")
            cv2.imwrite(filename, frame)
            smart_count += 1
            print(f"[智能 #{smart_count}] 已保存: {filename}")
        elif key == ord('r'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(regular_dir, f"regular_{ts}.jpg")
            cv2.imwrite(filename, frame)
            regular_count += 1
            print(f"[普通 #{regular_count}] 已保存: {filename}")
        elif key == ord('n'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(neg_dir, f"neg_{ts}.jpg")
            cv2.imwrite(filename, frame)
            neg_count += 1
            print(f"[背景 #{neg_count}] 已保存: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"采集结束。智能: {smart_count} 张, 普通: {regular_count} 张, 背景: {neg_count} 张")


if __name__ == "__main__":
    main()

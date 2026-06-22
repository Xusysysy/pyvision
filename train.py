"""
智能眼镜检测模型训练脚本 — train.py

使用 YOLO11n 训练单类别目标检测模型，并导出 ONNX 格式。
适用于树莓派 4B 部署。

用法:
  python train.py                          # 默认参数训练
  python train.py --epochs 150 --imgsz 416 # 自定义参数
  python train.py --export-only best.pt    # 仅导出已有模型

依赖:
  pip install ultralytics
"""

import argparse
import os
from pathlib import Path


def train(args):
    from ultralytics import YOLO

    data_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dataset", "data.yaml")

    if not os.path.exists(data_yaml):
        print(f"错误: 数据集配置文件不存在: {data_yaml}")
        print("请先运行: python prepare_dataset.py")
        return None

    print(f"数据配置: {data_yaml}")
    print(f"模型:     YOLO11n")
    print(f"轮数:     {args.epochs}")
    print(f"输入尺寸: {args.imgsz}")
    print(f"批大小:   {args.batch}")

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        patience=args.patience,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        # 单类检测 - 简化 Anchor
        single_cls=True,
        # 数据增强 - 适配小数据集
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n训练完成。最佳模型: {best_pt}")
        return str(best_pt)
    else:
        print("\n训练完成但未找到 best.pt")
        return None


def export_onnx(model_path: str, imgsz: int = 320):
    from ultralytics import YOLO

    print(f"\n导出 ONNX: {model_path} → smart_glasses.onnx (imgsz={imgsz})")
    model = YOLO(model_path)

    success = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        half=False,
    )

    if success:
        onnx_path = model_path.replace(".pt", ".onnx")
        target = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "smart_glasses.onnx")
        if os.path.abspath(onnx_path) != os.path.abspath(target):
            import shutil
            shutil.copy2(onnx_path, target)
            print(f"ONNX 已复制到: {target}")


def main():
    parser = argparse.ArgumentParser(description="训练智能眼镜检测模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=320, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--device", type=str, default=None, help="设备 (cpu / 0)")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--name", type=str, default="smart_glasses", help="训练名称")
    parser.add_argument("--export-only", type=str, default=None, help="仅导出已有 .pt 模型")
    parser.add_argument("--no-export", action="store_true", help="训练后不导出 ONNX")
    args = parser.parse_args()

    if args.export_only:
        export_onnx(args.export_only, args.imgsz)
        return

    best_pt = train(args)

    if best_pt and not args.no_export:
        export_onnx(best_pt, args.imgsz)
        print("\n完成! 部署文件: smart_glasses.onnx")


if __name__ == "__main__":
    main()

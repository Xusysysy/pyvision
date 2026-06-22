# pyvision

> GitHub: https://github.com/Xusysysy/pyvision.git

摄像头调试 + 智能眼镜检测工具。

## 功能

- **智能眼镜检测**：轻量 ONNX 模型，树莓派 4B 可部署
- **基础图像处理**：灰度、边缘检测、模糊、锐化、亮度/对比度调节
- **CNN 目标检测**：YOLO (.pt) / ONNX 模型实时推理
- **快照与录制**：截图、录像保存

## 训练自定义模型

```bash
# 1. 采集数据 (智能眼镜 + 普通眼镜 + 空桌面)
python collect_data.py --camera 0
#  P/Space=智能眼镜  R=普通眼镜  N=空桌面

# 2. 用 labelImg 框出眼镜并标注类别
pip install labelImg
labelImg dataset/images/smart     # 标 class 0 (smart_glasses)
labelImg dataset/images/regular   # 标 class 1 (regular_glasses)

# 3. 整理数据集
python prepare_dataset.py

# 4. 训练 (二分类: 智能 vs 普通眼镜)
python train.py --epochs 100 --imgsz 320

# 产出: smart_glasses.onnx
```

## 依赖

```bash
pip install opencv-python Pillow numpy

# CNN 推理
pip install onnxruntime        # ONNX 推理（用于智能眼镜检测）
pip install ultralytics        # YOLO 训练 + 推理
```

## 用法

```bash
python camera_debugger.py [--camera 0] [--width 640] [--height 480]
```

## 树莓派 4B 部署

```bash
pip install opencv-python onnxruntime Pillow numpy
scp camera_debugger.py smart_glasses.onnx pi@raspberrypi:~/pyvision/
python camera_debugger.py
```

## 打包

```bash
build.bat
```

## License

MIT

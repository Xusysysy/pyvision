# pyvision

> GitHub: https://github.com/Xusysysy/pyvision.git

摄像头调试工具 — 支持多种图像处理与 CNN 目标检测的实时调试工具。

## 功能

- **基础图像处理**：灰度、边缘检测、模糊、锐化、亮度/对比度调节
- **CNN 目标检测**：YOLO (.pt) / ONNX 模型实时推理
- **快照与录制**：截图、录像保存
- **DPI 感知**：Windows 高清屏适配

## 依赖

```bash
# 基础
pip install opencv-python Pillow numpy

# CNN 推理（二选一）
pip install ultralytics    # YOLO .pt 模型
pip install onnxruntime    # ONNX 模型
```

## 用法

```bash
python camera_debugger.py [--camera 0] [--width 640] [--height 480]
```

## 打包

```bash
build.bat
```

## License

MIT

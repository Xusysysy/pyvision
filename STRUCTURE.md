# STRUCTURE.md

## 项目结构

```
pyvision/
├── camera_debugger.py          # 主程序（图像处理 + GUI）
├── camera_debugger.spec        # PyInstaller 打包配置
├── build.bat                   # 一键打包脚本
├── yolov8n.pt                  # YOLO 预训练模型
├── CLAUDE.md                   # AI 编码规范
├── README.md                   # 项目说明
├── STRUCTURE.md                # 本文件
├── snapshots/                  # 截图保存目录
├── dist/                       # 打包输出
│   └── camera_debugger/
│       └── camera_debugger.exe
└── build/                      # 打包中间产物
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `camera_debugger.py` | 单文件主程序，包含图像处理处理器、GUI、录制、检测逻辑 |
| `FrameProcessor` | 图像处理抽象基类（灰度、边缘、模糊、锐化等） |
| `CNNProcessor` | YOLO/ONNX 模型推理处理器 |

## 依赖关系

```
opencv-python ← 主程序
Pillow        ← GUI 图像显示
numpy         ← 数值计算
ultralytics   ← YOLO 推理（可选）
onnxruntime   ← ONNX 推理（可选）
tkinter       ← GUI 框架（系统内置）
```

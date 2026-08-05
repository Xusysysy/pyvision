# pyvision

> GitHub: https://github.com/Xusysysy/pyvision.git

摄像头调试 + 智能眼镜识别工具集，包含两个独立应用：

- **camera_debugger**：摄像头调试工具（预览、偏移校准、拍照/录制、CNN 检测）
- **trainer**：训练工作台（数据采集 + 数据集准备 + 模型训练，全流程 GUI，零标注）

## 功能

### camera_debugger 调试工具

- **摄像头预览**：多摄像头切换、分辨率设置（预设/自定义）、水平镜像
- **偏移校准**：水平/竖直滑块平移画面，中心十字线对照，用于校正摄像头安装偏差
- **CNN 目标检测**：YOLO (.pt) / ONNX 模型实时推理（默认 `smart_glasses.onnx`）
- **基础图像处理**：灰度、边缘检测（Canny）
- **快照与录制**：一键截图、视频录制
- **设置持久化**：全部调整自动保存到 `settings.json`，启动自动恢复

### trainer 训练工作台

采用 **YOLO11 图像分类**，无需标注边界框，图片按类别存放即训练样本。

- **1. 采集数据**：摄像头预览（默认最大分辨率），三类一键保存（智能眼镜/普通眼镜/空桌面）
- **2. 准备数据**：自动按比例划分训练集/验证集
- **3. 训练模型**：参数可调（轮数/输入尺寸/批大小/设备），实时日志，同时输出 `.pt` 和 `.onnx` 分类模型

## 训练自定义模型

推荐使用 `trainer` 工作台完成全流程（旧脚本 `collect_data.py` / `prepare_dataset.py` / `train.py` 已并入其中）：

```bash
# 启动训练工作台
python trainer.py
```

1. **采集数据**：切换类别（快捷键 `1/2/3`），对准摄像头按 `Space` 保存当前帧
   - 智能眼镜 = 带摄像头/电池/按钮；普通眼镜 = 仅镜框镜片；空桌面 = 无眼镜背景
   - 建议每类采集 100+ 张，覆盖不同角度/距离/光照
2. **准备数据**：设置验证集比例（默认 0.2），点击"重新划分数据集"
3. **训练模型**：点击"开始训练"，完成后输出 `smart_glasses_cls.pt` 和 `smart_glasses_cls.onnx` 到数据目录（打包版为 `~/pyvision_dataset`，源码运行为项目 `dataset/`）

> 注意：训练需要联网下载基础模型 `yolo11n-cls.pt`（也可手动放入应用目录）。

## 依赖

```bash
pip install opencv-python Pillow numpy

# 训练 + CNN 推理
pip install ultralytics          # YOLO 训练 + 推理
pip install onnxruntime          # ONNX 推理（用于 smart_glasses.onnx 检测）
pip install onnx                 # ONNX 导出（trainer 同时输出 .pt 和 .onnx）
```

## 用法

```bash
# 调试工具（--width 0 表示自动使用摄像头最大分辨率）
python camera_debugger.py [--camera 0] [--width 0] [--height 0]

# 训练工作台
python trainer.py
```

## 打包

```bash
build.bat
# 输出: dist/camera_debugger/camera_debugger.exe
#       dist/trainer/trainer.exe
```

## 树莓派 4B 部署（检测模型）

```bash
pip install opencv-python onnxruntime Pillow numpy
scp camera_debugger.py smart_glasses.onnx pi@raspberrypi:~/pyvision/
python camera_debugger.py
```

## License

MIT

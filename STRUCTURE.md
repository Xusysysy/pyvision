# STRUCTURE.md

## 项目结构

```
pyvision/
├── camera_debugger.py          # 主程序（图像处理 + GUI + CNN 推理）
├── trainer.py                  # 训练工作台（采集 + 准备 + 训练，YOLO 分类，零标注）
├── collect_data.py             # 数据采集工具（采集训练图片）
├── prepare_dataset.py          # 数据集准备工具（整理标注文件）
├── train.py                    # 模型训练脚本（YOLO11n → ONNX）
├── camera_debugger.spec        # camera_debugger 打包配置
├── trainer.spec                # trainer 打包配置
├── build.bat                   # 一键打包脚本（两个 exe）
├── smart_glasses.onnx          # 智能眼镜检测 ONNX 模型（训练产出）
├── CLAUDE.md                   # AI 编码规范
├── README.md                   # 项目说明
├── STRUCTURE.md                # 本文件
├── dataset/                    # 训练数据根目录（trainer 使用）
│   ├── classes.json            # 分类标签配置（内部名称 + 显示名称，trainer 可编辑）
│   ├── datasets/               # 多数据集管理（trainer 第 0 步）
│   │   └── <数据集名>/          # 每个数据集独立目录（raw/train/val + 训练产物）
│   │       ├── raw/            # 采集暂存区（smart_glasses/regular_glasses/negative）
│   │       ├── train/          # 训练集（分类目录格式）
│   │       └── val/            # 验证集（分类目录格式）
│   ├── data.yaml               # 数据集配置（旧检测格式）
│   ├── images/                 # 图片（旧检测格式，train/val）
│   └── labels/                 # YOLO 标注（旧检测格式，train/val）
├── snapshots/                  # 截图保存目录
├── dist/                       # 打包输出
└── build/                      # 打包中间产物
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `camera_debugger.py` | 单文件主程序，包含图像处理处理器、GUI、录制、检测逻辑 |
| `trainer.py` | 训练工作台 GUI：数据集管理（新建/重命名/删除/切换）+ 分类标签编辑（增删改数量与名称，同步模型类名）+ 数据采集（带预览）+ 数据集划分 + YOLO11-cls 分类训练 |
| `collect_data.py` | USB 摄像头数据采集，按键分类保存正/负样本 |
| `prepare_dataset.py` | 整理标注文件并划分训练/验证集 |
| `train.py` | YOLO11n 单类检测训练 + ONNX 导出 |
| `FrameProcessor` | 图像处理抽象基类（灰度、边缘、模糊、锐化等） |
| `CNNProcessor` | YOLO/ONNX 模型推理处理器，默认使用 smart_glasses.onnx |
| `Settings` | JSON 设置持久化（camera_debugger 使用） |

## 依赖关系

```
opencv-python ← 主程序
Pillow        ← GUI 图像显示
numpy         ← 数值计算
ultralytics   ← YOLO 推理（可选）
onnxruntime   ← ONNX 推理（可选）
tkinter       ← GUI 框架（系统内置）
```

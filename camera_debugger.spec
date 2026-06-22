# -*- mode: python ; coding: utf-8 -*-
import os

datas = []
for model_file in ['smart_glasses.onnx', 'yolov8n.pt']:
    if os.path.isfile(model_file):
        datas.append((model_file, '.'))
    else:
        print(f"[spec] 模型文件不存在，跳过: {model_file}")

binaries = []
hiddenimports = []

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 只收集 ultralytics 推理所需的子模块，而非 collect_all 全部
hiddenimports += collect_submodules('ultralytics')
datas += collect_data_files('ultralytics', include_py_files=False)

# 不需要 certifi（代码里已禁用 SSL 验证）

excludes = [
    'scipy',
    'matplotlib',
    'pandas',
    'seaborn',
    'IPython',
    'notebook',
    'jupyter',
    'pytest',
    'sphinx',
    'docutils',
    'tensorboard',
    'wandb',
    'mlflow',
    'cv2.cuda',
    'ultralytics.solutions',
    'ultralytics.hub',
    'ultralytics.data.explorer',
]

a = Analysis(
    ['camera_debugger.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='camera_debugger',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='camera_debugger',
)

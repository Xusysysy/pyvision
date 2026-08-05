# -*- mode: python ; coding: utf-8 -*-
import os

datas = []
for model_file in ['yolo11n-cls.pt', 'smart_glasses.onnx']:
    if os.path.isfile(model_file):
        datas.append((model_file, '.'))
    else:
        print(f"[spec] 模型文件不存在，跳过: {model_file}")

binaries = []
hiddenimports = []

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 只收集 ultralytics 所需的子模块
hiddenimports += collect_submodules('ultralytics')
datas += collect_data_files('ultralytics', include_py_files=False)

# ONNX 导出（ultralytics 动态导入）
hiddenimports += collect_submodules('onnx')

excludes = [
    'scipy',
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
    'ultralytics.data.explorer',
]

a = Analysis(
    ['trainer.py'],
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
    name='trainer',
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
    name='trainer',
)

"""
智能眼镜训练工作台 — trainer.py

整合数据采集 + 数据集准备 + 模型训练 (YOLO11 图像分类，零标注)。
单一 GUI 应用，全流程：
  1. 采集数据  (摄像头预览，按类别保存图片)
  2. 准备数据  (自动划分 train/val)
  3. 训练模型  (YOLO11-cls，输出 .pt 分类模型)

用法: python trainer.py
输出: <应用目录>/smart_glasses_cls.pt

依赖: pip install opencv-python Pillow numpy ultralytics
"""

import os
import sys
import queue
import shutil
import random
import multiprocessing
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    raise SystemExit("需要 tkinter (Linux 下: sudo apt install python3-tk)")

from PIL import Image, ImageTk

# 在 Windows 上启用 DPI 感知（必须在创建 Tk 实例之前）
if sys.platform == "win32":
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception:
        pass

# PyInstaller 打包后修复 SSL 证书路径（用于下载基础模型）
if getattr(sys, "frozen", False):
    import ssl
    import certifi
    ssl._create_default_verify_paths = lambda: None
    ssl._create_default_https_context = ssl._create_unverified_context

from camera_debugger import CameraManager, enumerate_cameras


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _data_dir() -> str:
    """数据根目录。打包后放在用户主目录，避免被 PyInstaller 重建 exe 目录时误删。"""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), "pyvision_dataset")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


APP_DIR = _app_dir()
DATA_ROOT = _data_dir()

CLASSES = ["smart_glasses", "regular_glasses", "negative"]
CLASS_LABELS = ["智能眼镜", "普通眼镜", "空桌面"]


def _list_images(directory: str) -> list[str]:
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(
        [str(f) for f in p.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file()]
    )


def _count_by_class(root: str) -> list[int]:
    return [len(_list_images(os.path.join(root, c))) for c in CLASSES]


def _prepare_dataset(val_ratio: float):
    """从 dataset/raw 划分数据到 dataset/train|val（YOLO 分类目录格式）"""
    raw_root = os.path.join(DATA_ROOT, "raw")
    total_raw = sum(len(_list_images(os.path.join(raw_root, c))) for c in CLASSES)
    if total_raw == 0:
        raise RuntimeError("原始数据为空，请先在第 1 步采集数据")

    random.seed(42)
    for cls in CLASSES:
        for split in ("train", "val"):
            d = os.path.join(DATA_ROOT, split, cls)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    for cls in CLASSES:
        imgs = _list_images(os.path.join(raw_root, cls))
        random.shuffle(imgs)
        n_val = int(len(imgs) * val_ratio)
        if len(imgs) > 1:
            n_val = min(max(1, n_val), len(imgs) - 1)
        for i, img in enumerate(imgs):
            split = "val" if i < n_val else "train"
            shutil.copy2(img, os.path.join(DATA_ROOT, split, cls, os.path.basename(img)))


def _ensure_raw_dirs():
    for c in CLASSES:
        os.makedirs(os.path.join(DATA_ROOT, "raw", c), exist_ok=True)


# ═══════════════════════════════════════════════
# 训练子进程
# ═══════════════════════════════════════════════

def _train_process(log_q: multiprocessing.Queue, cfg: dict):
    """在独立进程中执行 YOLO 分类训练，日志写入队列"""
    class _Stream:
        def __init__(self, q):
            self.q = q

        def write(self, s):
            if s:
                self.q.put(s)
            return len(s)

        def flush(self):
            pass

    # 先替换 stdout/stderr，再 import ultralytics（其 logger 会绑定当前 stderr）
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Stream(log_q)
    sys.stderr = _Stream(log_q)

    try:
        from ultralytics import YOLO

        model = YOLO(cfg["base_model"])
        log_q.put(f"[TRAINER] 基础模型: {cfg['base_model']}\n")
        log_q.put(f"[TRAINER] 数据目录: {cfg['data_root']}\n")
        log_q.put(f"[TRAINER] epochs={cfg['epochs']} imgsz={cfg['imgsz']} "
                  f"batch={cfg['batch']} device={cfg['device']}\n")

        results = model.train(
            data=cfg["data_root"],
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=cfg["device"] or None,
            workers=cfg["workers"],
            patience=cfg["patience"],
            name=cfg["run_name"],
            plots=False,
            verbose=True,
        )

        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            out_path = os.path.join(cfg["out_dir"], cfg["out_name"])
            shutil.copy2(best_pt, out_path)
            log_q.put(f"\n[TRAINER] 模型已保存: {out_path}\n")

            # 导出 ONNX（同名 .onnx 输出到同一目录）
            try:
                log_q.put("[TRAINER] 导出 ONNX...\n")
                export_model = YOLO(best_pt)
                export_model.export(format="onnx", imgsz=cfg["imgsz"],
                                    simplify=True, opset=12)
                onnx_src = str(best_pt).replace(".pt", ".onnx")
                onnx_out = os.path.splitext(out_path)[0] + ".onnx"
                if os.path.isfile(onnx_src):
                    shutil.copy2(onnx_src, onnx_out)
                    log_q.put(f"[TRAINER] ONNX 已保存: {onnx_out}\n")
                else:
                    log_q.put("[TRAINER] ONNX 导出未生成文件\n")
            except ImportError:
                log_q.put("[TRAINER] ONNX 导出失败: 未安装 onnx 包，"
                          "请运行 pip install onnx onnxruntime 后重试\n")
            except Exception as e:
                log_q.put(f"[TRAINER] ONNX 导出失败: {e}\n")
        else:
            log_q.put("\n[TRAINER] 未找到 best.pt\n")
    except Exception as e:
        log_q.put(f"\n[TRAINER] 训练失败: {e}\n")
    finally:
        log_q.put("[TRAINER] DONE\n")
        sys.stdout, sys.stderr = old_out, old_err


# ═══════════════════════════════════════════════
# GUI 主界面
# ═══════════════════════════════════════════════

class TrainerGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.camera: CameraManager | None = None
        self._current_frame = None
        self._running = True
        self._camera_list: list[dict] = []
        self.class_index = 0
        self.counts = [0, 0, 0]
        self.offset_x = 0
        self.offset_y = 0

        self.train_proc: multiprocessing.Process | None = None
        self.log_q: multiprocessing.Queue = multiprocessing.Queue()

        # 显示缓存
        self._disp_cache = (0, 0, 0, 0, 0, 0)
        self._tk_image = None

        _ensure_raw_dirs()
        self._setup_ui()
        self._init_camera()
        self._scan_cameras()
        self._update_stats()
        self._update_preview()

    # ───────────── UI 构建 ─────────────

    def _setup_ui(self):
        self.root.title("智能眼镜训练工作台")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1100, 720)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#1a1a2e")
        style.configure("Dark.TLabel", background="#1a1a2e", foreground="#e0e0e0",
                        font=("Consolas", 10))
        style.configure("Section.TLabel", background="#1a1a2e", foreground="#5a9bd5",
                        font=("Consolas", 10, "bold"))
        style.configure("Title.TLabel", background="#1a1a2e", foreground="#00d4aa",
                        font=("Consolas", 14, "bold"))
        style.configure("Dark.TButton", font=("Consolas", 10))
        style.configure("Dark.TCheckbutton", background="#1a1a2e", foreground="#e0e0e0",
                        font=("Consolas", 10))
        style.configure("Dark.TEntry", fieldbackground="#16213e", foreground="#e0e0e0",
                        font=("Consolas", 9))
        style.configure("Dark.TNotebook", background="#1a1a2e", borderwidth=0)

        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        header = ttk.Frame(main, style="Dark.TFrame")
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="智能眼镜训练工作台", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="  |  采集 → 准备 → 训练 (YOLO 分类，零标注)",
                  style="Dark.TLabel").pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(header, textvariable=self.status_var, style="Dark.TLabel").pack(side=tk.RIGHT)

        nb = ttk.Notebook(main, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)

        self.tab_collect = ttk.Frame(nb, style="Dark.TFrame")
        self.tab_prepare = ttk.Frame(nb, style="Dark.TFrame")
        self.tab_train = ttk.Frame(nb, style="Dark.TFrame")
        nb.add(self.tab_collect, text=" 1. 采集数据 ")
        nb.add(self.tab_prepare, text=" 2. 准备数据 ")
        nb.add(self.tab_train, text=" 3. 训练模型 ")

        self._build_collect_tab()
        self._build_prepare_tab()
        self._build_train_tab()

        # 底部状态栏
        status_frame = ttk.Frame(main, style="Dark.TFrame")
        status_frame.pack(fill=tk.X, pady=(8, 0))
        self.footer_var = tk.StringVar(value="快捷键: 1/2/3 选择类别 | Space 保存 | Q 退出")
        ttk.Label(status_frame, textvariable=self.footer_var,
                  style="Dark.TLabel").pack(side=tk.LEFT)

        self.root.bind("<q>", lambda e: self._on_close())
        self.root.bind("<Q>", lambda e: self._on_close())
        self.root.bind("<space>", lambda e: self._save_frame())
        self.root.bind("1", lambda e: self._select_class(0))
        self.root.bind("2", lambda e: self._select_class(1))
        self.root.bind("3", lambda e: self._select_class(2))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_collect_tab(self):
        tab = self.tab_collect

        body = ttk.Frame(tab, style="Dark.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左：摄像头预览
        video = ttk.Frame(body, style="Dark.TFrame")
        video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(video, bg="#0d0d0d", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 右：控制面板
        ctrl = ttk.Frame(body, style="Dark.TFrame", width=300)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        ctrl.pack_propagate(False)

        ttk.Label(ctrl, text="摄像头", style="Section.TLabel").pack(anchor=tk.W, pady=(4, 4))
        cam_row = ttk.Frame(ctrl, style="Dark.TFrame")
        cam_row.pack(fill=tk.X, pady=(0, 4))
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(cam_row, textvariable=self.camera_var,
                                         state="readonly", width=18)
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_switch)
        ttk.Button(cam_row, text="刷新", width=5,
                   command=self._scan_cameras).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(ctrl, text="选择类别", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.class_buttons = []
        for i, label in enumerate(CLASS_LABELS):
            btn = tk.Button(ctrl, text=f"{label} ({i + 1})", font=("Consolas", 11, "bold"),
                            relief="flat", cursor="hand2", height=1,
                            command=lambda idx=i: self._select_class(idx))
            btn.pack(fill=tk.X, pady=2)
            self.class_buttons.append(btn)

        self._select_class(0)

        self.save_btn = tk.Button(
            ctrl, text="保存当前帧 (Space)", font=("Consolas", 12, "bold"),
            bg="#00d4aa", fg="#1a1a2e", activebackground="#00b894",
            relief="flat", cursor="hand2", height=2,
            command=self._save_frame
        )
        self.save_btn.pack(fill=tk.X, pady=(8, 4))

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(ctrl, text="画面偏移 (采集校准)", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.offset_x_var = tk.IntVar(value=0)
        self.offset_y_var = tk.IntVar(value=0)

        hrow = ttk.Frame(ctrl, style="Dark.TFrame")
        hrow.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(hrow, text="水平:", style="Dark.TLabel").pack(side=tk.LEFT)
        ttk.Scale(hrow, from_=-300, to=300, variable=self.offset_x_var,
                  command=self._on_offset_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Entry(hrow, textvariable=self.offset_x_var, width=6).pack(side=tk.LEFT)

        vrow = ttk.Frame(ctrl, style="Dark.TFrame")
        vrow.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(vrow, text="竖直:", style="Dark.TLabel").pack(side=tk.LEFT)
        ttk.Scale(vrow, from_=-300, to=300, variable=self.offset_y_var,
                  command=self._on_offset_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Entry(vrow, textvariable=self.offset_y_var, width=6).pack(side=tk.LEFT)

        ttk.Button(ctrl, text="清零偏移", width=8,
                   command=self._reset_offset).pack(anchor=tk.W, pady=(2, 4))

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(ctrl, text="采集统计", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.stats_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.stats_var, style="Dark.TLabel",
                  justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Label(ctrl, text="\n提示: 智能眼镜 = 带摄像头/电池\n普通眼镜 = 仅镜框镜片\n空桌面 = 无眼镜背景",
                  style="Dark.TLabel", justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

    def _build_prepare_tab(self):
        tab = self.tab_prepare

        box = ttk.Frame(tab, style="Dark.TFrame")
        box.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        ttk.Label(box, text="数据集准备 (自动划分训练/验证集)",
                  style="Section.TLabel").pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(box, text="原始数据目录:", style="Dark.TLabel").pack(anchor=tk.W)
        self.data_root_label = ttk.Label(box, text=DATA_ROOT, style="Dark.TLabel",
                                         foreground="#5a9bd5")
        self.data_root_label.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(box, text="原始数据统计 (dataset/raw):", style="Dark.TLabel").pack(anchor=tk.W)
        self.raw_stats_var = tk.StringVar(value="--")
        ttk.Label(box, textvariable=self.raw_stats_var, style="Dark.TLabel").pack(anchor=tk.W, pady=(0, 8))

        row = ttk.Frame(box, style="Dark.TFrame")
        row.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(row, text="验证集比例:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.val_ratio_var = tk.StringVar(value="0.2")
        ttk.Entry(row, textvariable=self.val_ratio_var, width=6).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(row, text="  (0.0-1.0，例如 0.2 = 20% 验证)", style="Dark.TLabel").pack(side=tk.LEFT)

        self.prepare_btn = ttk.Button(box, text="重新划分数据集", style="Dark.TButton",
                                      command=self._do_prepare)
        self.prepare_btn.pack(anchor=tk.W, pady=(0, 12))

        ttk.Separator(box, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(box, text="划分结果", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))
        self.prepared_stats_var = tk.StringVar(value="--")
        ttk.Label(box, textvariable=self.prepared_stats_var, style="Dark.TLabel",
                  justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Label(box, text="\n说明: 按类别目录直接作为训练样本，无需标注边界框。\n"
                            "划分会把 dataset/raw 下的图片按比例复制到 dataset/train 和 dataset/val。",
                  style="Dark.TLabel", justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))

    def _build_train_tab(self):
        tab = self.tab_train

        left = ttk.Frame(tab, style="Dark.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        ttk.Label(left, text="训练参数 (YOLO11 分类)", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 8))

        grid = ttk.Frame(left, style="Dark.TFrame")
        grid.pack(fill=tk.X)

        params = [
            ("训练轮数 epochs", "epochs_var", "100"),
            ("输入尺寸 imgsz", "imgsz_var", "224"),
            ("批大小 batch", "batch_var", "16"),
            ("设备 device (留空=自动)", "device_var", ""),
            ("数据加载线程 workers", "workers_var", "4"),
            ("早停 patience", "patience_var", "30"),
        ]
        self.param_vars = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(grid, text=label, style="Dark.TLabel").grid(
                row=i, column=0, sticky=tk.W, pady=3)
            v = tk.StringVar(value=default)
            self.param_vars[key] = v
            ttk.Entry(grid, textvariable=v, width=14).grid(row=i, column=1, sticky=tk.W, pady=3, padx=(8, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="基础模型 (需联网下载或已存在):", style="Dark.TLabel").pack(anchor=tk.W)
        self.base_var = tk.StringVar(value="yolo11n-cls.pt")
        ttk.Entry(left, textvariable=self.base_var, width=40).pack(fill=tk.X, pady=(2, 6))

        ttk.Label(left, text="输出模型文件名 (同时导出 .pt 和 .onnx):", style="Dark.TLabel").pack(anchor=tk.W)
        self.out_name_var = tk.StringVar(value="smart_glasses_cls.pt")
        ttk.Entry(left, textvariable=self.out_name_var, width=40).pack(fill=tk.X, pady=(2, 10))

        self.train_btn = tk.Button(
            left, text="开始训练", font=("Consolas", 12, "bold"),
            bg="#00d4aa", fg="#1a1a2e", activebackground="#00b894",
            relief="flat", cursor="hand2", height=2,
            command=self._start_training
        )
        self.train_btn.pack(fill=tk.X, pady=(0, 6))

        self.stop_btn = tk.Button(
            left, text="停止训练", font=("Consolas", 11),
            bg="#e74c3c", fg="#ffffff", activebackground="#c0392b",
            relief="flat", cursor="hand2", state="disabled",
            command=self._stop_training
        )
        self.stop_btn.pack(fill=tk.X)

        ttk.Label(left, text="\n数据集统计 (train/val):", style="Dark.TLabel").pack(anchor=tk.W, pady=(8, 0))
        self.train_stats_var = tk.StringVar(value="--")
        ttk.Label(left, textvariable=self.train_stats_var, style="Dark.TLabel",
                  justify=tk.LEFT).pack(anchor=tk.W)

        # 右：日志
        right = ttk.Frame(tab, style="Dark.TFrame", width=520)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 12), pady=12)
        right.pack_propagate(False)

        ttk.Label(right, text="训练日志", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 6))

        log_frame = ttk.Frame(right, style="Dark.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, bg="#0d0d0d", fg="#b8c4d0",
                                font=("Consolas", 9), wrap="none",
                                relief="flat", state="disabled")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                  command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    # ───────────── 摄像头 ─────────────

    def _init_camera(self):
        try:
            # width/height=0 表示自动使用摄像头支持的最大分辨率
            self.camera = CameraManager(0, 0, 0)
        except RuntimeError as e:
            self.camera = None
            self.status_var.set(f"无法打开摄像头: {e}")

    def _scan_cameras(self):
        cams = enumerate_cameras(max_id=6)
        self._camera_list = cams
        names = [c["name"] for c in cams] or ["未检测到摄像头"]
        self.camera_combo["values"] = names
        if self.camera is not None:
            idx = next((i for i, c in enumerate(cams) if c["id"] == self.camera.camera_id), 0)
            self.camera_combo.current(idx)

    def _on_camera_switch(self, event=None):
        sel = self.camera_combo.current()
        if sel < 0 or sel >= len(self._camera_list) or self.camera is None:
            return
        new_id = self._camera_list[sel]["id"]
        if new_id == self.camera.camera_id:
            return
        try:
            self.camera.switch(new_id)
            self.status_var.set(f"已切换到摄像头 {new_id}")
        except RuntimeError as e:
            messagebox.showerror("切换失败", str(e))

    # ───────────── 采集 ─────────────

    def _select_class(self, idx: int):
        self.class_index = idx
        colors = ["#00d4aa", "#f39c12", "#9b59b6"]
        for i, btn in enumerate(self.class_buttons):
            if i == idx:
                btn.config(bg=colors[idx], fg="#1a1a2e")
            else:
                btn.config(bg="#16213e", fg="#e0e0e0")

    def _save_frame(self):
        if self._current_frame is None:
            self.status_var.set("没有可保存的帧")
            return
        cls = CLASSES[self.class_index]
        d = os.path.join(DATA_ROOT, "raw", cls)
        os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{cls}_{ts}.jpg"
        path = os.path.join(d, name)
        frame_to_save = self._current_frame
        if self.offset_x or self.offset_y:
            frame_to_save = self._apply_offset(self._current_frame)
        ok = cv2.imwrite(path, frame_to_save)
        if ok:
            self.counts[self.class_index] += 1
            self.status_var.set(f"已保存: {name}")
            self._update_stats()
        else:
            self.status_var.set(f"保存失败: {path}")

    def _update_stats(self):
        self.counts = _count_by_class(os.path.join(DATA_ROOT, "raw"))
        self.stats_var.set(
            f"智能眼镜: {self.counts[0]} 张\n普通眼镜: {self.counts[1]} 张\n空桌面:   {self.counts[2]} 张"
        )

    # ───────────── 画面偏移 ─────────────

    def _on_offset_change(self, event=None):
        self.offset_x = int(self.offset_x_var.get())
        self.offset_y = int(self.offset_y_var.get())

    def _reset_offset(self):
        self.offset_x_var.set(0)
        self.offset_y_var.set(0)
        self._on_offset_change()

    def _apply_offset(self, frame: np.ndarray) -> np.ndarray:
        """水平/竖直平移画面，边缘用黑色填充"""
        if self.offset_x == 0 and self.offset_y == 0:
            return frame
        h, w = frame.shape[:2]
        m = np.float32([[1, 0, self.offset_x], [0, 1, self.offset_y]])
        return cv2.warpAffine(frame, m, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    # ───────────── 预览循环 ─────────────

    def _update_preview(self):
        if not self._running:
            return
        if self.camera:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                self._current_frame = frame
                self._show_frame(frame)
        self.root.after(16, self._update_preview)

    def _show_frame(self, frame: np.ndarray):
        if self.offset_x or self.offset_y:
            frame = self._apply_offset(frame)
        disp = frame.copy()
        h, w = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (w, 34), (30, 30, 30), -1)
        cv2.putText(disp, f"Class: {CLASSES[self.class_index]}  |  Smart:{self.counts[0]} "
                          f"Reg:{self.counts[1]}  Neg:{self.counts[2]}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        if self._disp_cache[:2] != (w, h) or self._disp_cache[2:4] != (cw, ch):
            scale = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            self._disp_cache = (w, h, cw, ch, nw, nh)

        _, _, _, _, nw, nh = self._disp_cache
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.BILINEAR)
        self._tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._tk_image)

    # ───────────── 数据准备 ─────────────

    def _do_prepare(self):
        try:
            val_ratio = float(self.val_ratio_var.get())
        except ValueError:
            messagebox.showerror("错误", "验证集比例必须是数字")
            return
        if not 0.0 < val_ratio < 1.0:
            messagebox.showerror("错误", "验证集比例必须在 0-1 之间")
            return

        raw_root = os.path.join(DATA_ROOT, "raw")
        total_raw = sum(len(_list_images(os.path.join(raw_root, c))) for c in CLASSES)
        if total_raw == 0:
            messagebox.showerror("错误", "原始数据为空，请先在第 1 步采集数据")
            return

        self.status_var.set("正在划分数据集...")
        self.root.update_idletasks()

        try:
            _prepare_dataset(val_ratio)
        except RuntimeError as e:
            messagebox.showerror("错误", str(e))
            return

        self._update_data_stats()
        self.status_var.set("数据集划分完成")
        messagebox.showinfo("完成", "数据集划分完成，可在第 3 步开始训练")

    def _update_data_stats(self):
        raw_counts = _count_by_class(os.path.join(DATA_ROOT, "raw"))
        self.raw_stats_var.set(
            f"智能: {raw_counts[0]} 张 | 普通: {raw_counts[1]} 张 | 空桌面: {raw_counts[2]} 张"
        )
        train_counts = _count_by_class(os.path.join(DATA_ROOT, "train"))
        val_counts = _count_by_class(os.path.join(DATA_ROOT, "val"))
        self.prepared_stats_var.set(
            f"训练集: 智能 {train_counts[0]} / 普通 {train_counts[1]} / 空桌面 {train_counts[2]}\n"
            f"验证集: 智能 {val_counts[0]} / 普通 {val_counts[1]} / 空桌面 {val_counts[2]}"
        )
        self.train_stats_var.set(
            f"训练集 {sum(train_counts)} 张 | 验证集 {sum(val_counts)} 张\n"
            f"(智能 {train_counts[0]}/{val_counts[0]} | 普通 {train_counts[1]}/{val_counts[1]} "
            f"| 空桌面 {train_counts[2]}/{val_counts[2]})"
        )

    # ───────────── 训练 ─────────────

    def _start_training(self):
        if self.train_proc and self.train_proc.is_alive():
            return

        train_root = os.path.join(DATA_ROOT, "train")
        total = sum(len(_list_images(os.path.join(train_root, c))) for c in CLASSES)
        if total == 0:
            messagebox.showerror("错误", "训练集为空，请先采集数据并划分数据集")
            return

        try:
            cfg = {
                "data_root": DATA_ROOT,
                "base_model": self.base_var.get().strip() or "yolo11n-cls.pt",
                "epochs": max(1, int(self.param_vars["epochs_var"].get())),
                "imgsz": max(32, int(self.param_vars["imgsz_var"].get())),
                "batch": max(1, int(self.param_vars["batch_var"].get())),
                "device": self.param_vars["device_var"].get().strip(),
                "workers": max(0, int(self.param_vars["workers_var"].get())),
                "patience": max(0, int(self.param_vars["patience_var"].get())),
                "run_name": "smart_glasses",
                "out_dir": DATA_ROOT,
                "out_name": self.out_name_var.get().strip() or "smart_glasses_cls.pt",
            }
        except ValueError as e:
            messagebox.showerror("错误", f"参数错误: {e}")
            return

        self.log_q = multiprocessing.Queue()
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

        self.train_proc = multiprocessing.Process(target=_train_process, args=(self.log_q, cfg))
        self.train_proc.start()
        self.train_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("训练进行中...")
        self._poll_log()

    def _stop_training(self):
        if self.train_proc and self.train_proc.is_alive():
            self.train_proc.terminate()
            self.train_proc.join(timeout=3)
            self.train_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set("已停止训练")
            self._append_log("\n[TRAINER] 训练已由用户停止\n")

    def _poll_log(self):
        self._drain_log()
        if self.train_proc and self.train_proc.is_alive():
            self.root.after(100, self._poll_log)
        else:
            self._drain_log()
            self.train_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set("训练已结束，模型位于应用目录")

    def _drain_log(self):
        try:
            while True:
                chunk = self.log_q.get_nowait()
                self._append_log(chunk)
        except (queue.Empty, ValueError, OSError):
            pass

    def _append_log(self, chunk: str):
        self.log_text.config(state="normal")
        # 处理 tqdm 的 \r 进度：覆盖最后一行
        parts = chunk.split("\r")
        for i, part in enumerate(parts):
            if not part:
                continue
            if i < len(parts) - 1 or chunk.endswith("\r"):
                if "\n" in part:
                    lines = part.split("\n")
                    for ln in lines:
                        if ln:
                            self._replace_last_line(ln)
                else:
                    self._replace_last_line(part)
            else:
                self.log_text.insert(tk.END, part)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _replace_last_line(self, text: str):
        idx = self.log_text.index("end-1c")
        line_start = self.log_text.index("end-1c linestart")
        self.log_text.delete(line_start, idx)
        self.log_text.insert(tk.END, text)

    # ───────────── 关闭 ─────────────

    def _on_close(self):
        self._running = False
        if self.train_proc and self.train_proc.is_alive():
            self.train_proc.terminate()
            self.train_proc.join(timeout=2)
        if self.camera:
            self.camera.release()
        self.root.destroy()


# ═══════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════

def main():
    multiprocessing.freeze_support()
    root = tk.Tk()
    root.geometry("1280x800")
    app = TrainerGUI(root)
    try:
        root.mainloop()
    finally:
        if app.camera:
            app.camera.stop()


if __name__ == "__main__":
    main()

"""
智能眼镜训练工作台 — trainer.py

整合数据集管理 + 分类标签编辑 + 数据采集 + 数据集准备 + 模型训练 (YOLO11 图像分类，零标注)。
单一 GUI 应用，全流程：
  0. 管理数据集 (新建/重命名/删除/切换，每个数据集独立生成不同模型)
     编辑分类标签 (增删改数量与名称，同步模型类名/目录)
  1. 采集数据  (摄像头预览，按类别保存图片)
  2. 准备数据  (自动划分 train/val)
  3. 训练模型  (YOLO11-cls，输出 .pt 分类模型)

用法: python trainer.py
输出: <应用目录>/smart_glasses_cls.pt

依赖: pip install opencv-python Pillow numpy ultralytics
"""

import os
import re
import sys
import json
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
    from tkinter import ttk, messagebox, filedialog, simpledialog
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

# 每个数据集独立目录: <DATA_ROOT>/datasets/<名称>/{raw,train,val}
DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")


def _dataset_dir(name: str) -> str:
    return os.path.join(DATASETS_DIR, name)


def _list_datasets() -> list[str]:
    p = Path(DATASETS_DIR)
    if not p.is_dir():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def _migrate_legacy_dataset():
    """首次运行: 把旧版顶层 raw/train/val 迁入 datasets/default，避免数据丢失"""
    if os.path.isdir(DATASETS_DIR):
        return
    os.makedirs(DATASETS_DIR, exist_ok=True)
    default_dir = _dataset_dir("default")
    os.makedirs(default_dir, exist_ok=True)
    for sub in ("raw", "train", "val"):
        src = os.path.join(DATA_ROOT, sub)
        if os.path.isdir(src):
            shutil.move(src, os.path.join(default_dir, sub))


DEFAULT_CLASSES = ["smart_glasses", "regular_glasses", "negative"]
DEFAULT_CLASS_LABELS = ["智能眼镜", "普通眼镜", "空桌面"]
CLASSES_FILE = os.path.join(DATA_ROOT, "classes.json")


def _load_classes() -> tuple[list[str], list[str]]:
    """从 classes.json 读取分类配置 (内部名称 + 显示名称)，不存在则用默认值"""
    if os.path.isfile(CLASSES_FILE):
        try:
            with open(CLASSES_FILE, encoding="utf-8") as f:
                d = json.load(f)
            classes = [str(c) for c in d.get("classes", [])]
            labels = [str(c) for c in d.get("labels", [])]
            if classes and len(classes) == len(labels):
                return classes, labels
        except Exception:
            pass
    return list(DEFAULT_CLASSES), list(DEFAULT_CLASS_LABELS)


CLASSES, CLASS_LABELS = _load_classes()


def _save_classes(classes: list[str], labels: list[str]):
    os.makedirs(DATA_ROOT, exist_ok=True)
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "labels": labels}, f,
                  ensure_ascii=False, indent=2)


def _rename_class_dir(old: str, new: str):
    """把所有数据集下 raw/train/val 的类别目录重命名 (同步模型类名)"""
    for ds in _list_datasets():
        for split in ("raw", "train", "val"):
            src = os.path.join(_dataset_dir(ds), split, old)
            if os.path.isdir(src):
                os.rename(src, os.path.join(_dataset_dir(ds), split, new))


def _add_class_dir(cls: str):
    for ds in _list_datasets():
        for split in ("raw", "train", "val"):
            os.makedirs(os.path.join(_dataset_dir(ds), split, cls), exist_ok=True)


def _remove_class_dir(cls: str):
    for ds in _list_datasets():
        for split in ("raw", "train", "val"):
            shutil.rmtree(os.path.join(_dataset_dir(ds), split, cls), ignore_errors=True)


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


def _prepare_dataset(ds_dir: str, val_ratio: float):
    """从 <ds_dir>/raw 划分数据到 <ds_dir>/train|val（YOLO 分类目录格式）"""
    raw_root = os.path.join(ds_dir, "raw")
    total_raw = sum(len(_list_images(os.path.join(raw_root, c))) for c in CLASSES)
    if total_raw == 0:
        raise RuntimeError("原始数据为空，请先在第 1 步采集数据")

    random.seed(42)
    for cls in CLASSES:
        for split in ("train", "val"):
            d = os.path.join(ds_dir, split, cls)
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
            shutil.copy2(img, os.path.join(ds_dir, split, cls, os.path.basename(img)))


def _ensure_raw_dirs(ds_dir: str):
    for c in CLASSES:
        os.makedirs(os.path.join(ds_dir, "raw", c), exist_ok=True)


# ═══════════════════════════════════════════════
# 训练子进程
# ═══════════════════════════════════════════════

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_NOISE_PREFIXES = ("Ultralytics ", "Predict: ", "Validate: ", "Visualize: ",
                    "Results saved to ", "requirements:")
_NOISE_SUBSTRINGS = ("summary (fused)", "Failed to inspect Python interpreter",
                     "Caused by: Querying Python", "WARNING Retry", "uv pip install")


def _is_noise_line(line: str) -> bool:
    """过滤 ultralytics 横幅/AutoUpdate 等对用户无价值的日志行"""
    s = line.strip()
    if not s:
        return False
    if s.startswith(_NOISE_PREFIXES):
        return True
    return any(t in s for t in _NOISE_SUBSTRINGS)


def _train_process(log_q: multiprocessing.Queue, cfg: dict):
    """在独立进程中执行 YOLO 分类训练，日志写入队列"""
    # 打包环境下 sys.executable 是 trainer.exe，ultralytics AutoUpdate 会用 uv
    # 把 exe 当作 Python 解释器启动探测，导致训练结束时弹出新的 trainer 窗口；禁用自动安装
    os.environ["YOLO_AUTOINSTALL"] = "0"

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
            base = cfg["out_name"]
            out_pt = os.path.join(cfg["out_dir"], base + ".pt")
            out_onnx = os.path.join(cfg["out_dir"], base + ".onnx")
            shutil.copy2(best_pt, out_pt)
            log_q.put(f"\n[TRAINER] 模型已保存: {out_pt}\n")

            # 导出 ONNX（同名 .onnx 输出到同一目录）
            try:
                log_q.put("[TRAINER] 导出 ONNX...\n")
                export_model = YOLO(best_pt)
                export_model.export(format="onnx", imgsz=cfg["imgsz"],
                                    simplify=True, opset=12)
                onnx_src = str(best_pt).replace(".pt", ".onnx")
                if os.path.isfile(onnx_src):
                    shutil.copy2(onnx_src, out_onnx)
                    log_q.put(f"[TRAINER] ONNX 已保存: {out_onnx}\n")
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
        self.counts = [0] * len(CLASSES)
        self.offset_x = 0
        self.offset_y = 0

        self.train_proc: multiprocessing.Process | None = None
        self.log_q: multiprocessing.Queue = multiprocessing.Queue()
        self.dataset_name = "default"

        # 显示缓存
        self._disp_cache = (0, 0, 0, 0, 0, 0)
        self._tk_image = None

        _migrate_legacy_dataset()
        _ensure_raw_dirs(self.dataset_dir)
        self._setup_ui()
        self._refresh_datasets()
        self._init_camera()
        self._scan_cameras()
        self._update_preview()

    @property
    def dataset_dir(self) -> str:
        """当前数据集根目录 (含 raw/train/val)"""
        return _dataset_dir(self.dataset_name)

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

        self.tab_dataset = ttk.Frame(nb, style="Dark.TFrame")
        self.tab_collect = ttk.Frame(nb, style="Dark.TFrame")
        self.tab_prepare = ttk.Frame(nb, style="Dark.TFrame")
        self.tab_train = ttk.Frame(nb, style="Dark.TFrame")
        nb.add(self.tab_dataset, text=" 0. 数据集管理 ")
        nb.add(self.tab_collect, text=" 1. 采集数据 ")
        nb.add(self.tab_prepare, text=" 2. 准备数据 ")
        nb.add(self.tab_train, text=" 3. 训练模型 ")

        self._build_dataset_tab()
        self._build_collect_tab()
        self._build_prepare_tab()
        self._build_train_tab()

        # 底部状态栏
        status_frame = ttk.Frame(main, style="Dark.TFrame")
        status_frame.pack(fill=tk.X, pady=(8, 0))
        self.footer_var = tk.StringVar(value="快捷键: 1-9 选择类别 | Space 保存 | Q 退出")
        ttk.Label(status_frame, textvariable=self.footer_var,
                  style="Dark.TLabel").pack(side=tk.LEFT)

        self.root.bind("<q>", lambda e: self._on_close())
        self.root.bind("<Q>", lambda e: self._on_close())
        self.root.bind("<space>", lambda e: self._save_frame())
        for i in range(1, 10):
            self.root.bind(str(i), lambda e, idx=i - 1: self._select_class(idx))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_dataset_tab(self):
        tab = self.tab_dataset

        box = ttk.Frame(tab, style="Dark.TFrame")
        box.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        ttk.Label(box, text="数据集管理 (每个数据集独立采集/训练，生成不同模型)",
                  style="Section.TLabel").pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(box, text="当前数据集:", style="Dark.TLabel").pack(anchor=tk.W)
        row = ttk.Frame(box, style="Dark.TFrame")
        row.pack(fill=tk.X, pady=(2, 8))
        self.dataset_combo = ttk.Combobox(row, state="readonly", width=30)
        self.dataset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_switch)
        ttk.Button(row, text="刷新", width=6,
                   command=self._refresh_datasets).pack(side=tk.LEFT, padx=(4, 0))

        btns = ttk.Frame(box, style="Dark.TFrame")
        btns.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btns, text="新建数据集", width=12,
                   command=self._new_dataset).pack(side=tk.LEFT)
        ttk.Button(btns, text="重命名", width=10,
                   command=self._rename_dataset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text="删除数据集", width=12,
                   command=self._delete_dataset).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(box, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(box, text="数据集统计", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))
        self.ds_stats_var = tk.StringVar(value="--")
        ttk.Label(box, textvariable=self.ds_stats_var, style="Dark.TLabel",
                  justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Separator(box, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(box, text="分类标签 (数量/名称)", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))
        self.class_summary_var = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.class_summary_var, style="Dark.TLabel",
                  justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Button(box, text="编辑分类标签...", width=14,
                   command=self._edit_labels).pack(anchor=tk.W, pady=(4, 0))
        self._update_class_summary()

        ttk.Label(box, text="\n数据集目录: " + DATASETS_DIR + "\n"
                            "采集/准备/训练均作用于当前选择的数据集。\n"
                            "训练产出的模型 (.pt/.onnx) 保存在对应数据集目录下。",
                  style="Dark.TLabel", justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))

    # ───────────── 数据集管理 ─────────────

    def _refresh_datasets(self, select: str | None = None):
        names = _list_datasets()
        if not names:
            _ensure_raw_dirs(_dataset_dir("default"))
            names = _list_datasets()
        if select and select in names:
            self.dataset_name = select
        elif self.dataset_name not in names:
            self.dataset_name = names[0]
        self.dataset_combo["values"] = names
        self.dataset_combo.current(names.index(self.dataset_name))
        self._on_dataset_changed()

    def _on_dataset_switch(self, event=None):
        sel = self.dataset_combo.current()
        names = _list_datasets()
        if 0 <= sel < len(names) and names[sel] != self.dataset_name:
            self.dataset_name = names[sel]
            self._on_dataset_changed()

    def _on_dataset_changed(self):
        _ensure_raw_dirs(self.dataset_dir)
        self._update_stats()
        self._update_data_stats()
        self._update_dataset_stats()
        self.data_root_label.config(text=self.dataset_dir)
        self.status_var.set(f"当前数据集: {self.dataset_name}")

    def _validate_ds_name(self, name: str) -> str | None:
        name = name.strip()
        if not name:
            messagebox.showwarning("提示", "名称不能为空")
            return None
        if any(c in name for c in '/\\:*?"<>|'):
            messagebox.showerror("错误", "名称不能包含特殊字符 / \\ : * ? \" < > |")
            return None
        return name

    def _new_dataset(self):
        name = simpledialog.askstring("新建数据集", "数据集名称 (例如: 室内检测):",
                                      parent=self.root)
        if not name:
            return
        name = self._validate_ds_name(name)
        if not name:
            return
        if name in _list_datasets():
            messagebox.showerror("错误", f"数据集 \"{name}\" 已存在")
            return
        _ensure_raw_dirs(_dataset_dir(name))
        self._refresh_datasets(select=name)
        self.status_var.set(f"已创建数据集: {name}")

    def _rename_dataset(self):
        if not self.dataset_name:
            return
        name = simpledialog.askstring("重命名数据集",
                                      f"将 \"{self.dataset_name}\" 重命名为:",
                                      initialvalue=self.dataset_name, parent=self.root)
        if not name:
            return
        name = self._validate_ds_name(name)
        if not name:
            return
        if name == self.dataset_name:
            return
        if name in _list_datasets():
            messagebox.showerror("错误", f"数据集 \"{name}\" 已存在")
            return
        os.rename(_dataset_dir(self.dataset_name), _dataset_dir(name))
        self._refresh_datasets(select=name)
        self.status_var.set(f"已重命名为: {name}")

    def _delete_dataset(self):
        name = self.dataset_name
        if not name:
            return
        n = sum(len(_list_images(os.path.join(self.dataset_dir, "raw", c))) for c in CLASSES)
        if not messagebox.askyesno("删除数据集",
                                   f"确定删除数据集 \"{name}\"？\n"
                                   f"将删除其全部数据 ({n} 张图片) 及训练模型，不可恢复。",
                                   parent=self.root):
            return
        shutil.rmtree(self.dataset_dir, ignore_errors=True)
        self._refresh_datasets()
        self.status_var.set(f"已删除数据集: {name}")

    def _update_dataset_stats(self):
        raw = _count_by_class(os.path.join(self.dataset_dir, "raw"))
        tr = _count_by_class(os.path.join(self.dataset_dir, "train"))
        va = _count_by_class(os.path.join(self.dataset_dir, "val"))
        self.ds_stats_var.set(
            f"数据集: {self.dataset_name}  ({self.dataset_dir})\n"
            f"原始: " + ", ".join(f"{CLASS_LABELS[i]} {raw[i]}" for i in range(len(CLASSES))) + "\n"
            f"训练: {sum(tr)} 张 | 验证: {sum(va)} 张"
        )

    # ───────────── 分类标签编辑 ─────────────

    def _update_class_summary(self):
        self.class_summary_var.set(
            f"共 {len(CLASSES)} 类\n" +
            "\n".join(f"{i + 1}. {CLASS_LABELS[i]} ({CLASSES[i]})" for i in range(len(CLASSES)))
        )

    def _validate_class_key(self, key: str, exclude: int = -1) -> str | None:
        key = key.strip()
        if not key:
            messagebox.showwarning("提示", "内部名称不能为空")
            return None
        if any(c in key for c in '/\\:*?"<>|'):
            messagebox.showerror("错误", "内部名称不能包含特殊字符 / \\ : * ? \" < > |")
            return None
        for i, k in enumerate(CLASSES):
            if i != exclude and k == key:
                messagebox.showerror("错误", f"内部名称 \"{key}\" 已存在")
                return None
        return key

    def _edit_labels(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("编辑分类标签")
        dlg.configure(bg="#1a1a2e")
        dlg.geometry("520x420")
        dlg.transient(self.root)
        dlg.grab_set()

        frame = ttk.Frame(dlg, style="Dark.TFrame")
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        ttk.Label(frame, text="分类标签 (内部名称 = 目录/模型类名; 显示名称 = 界面文字)",
                  style="Section.TLabel").pack(anchor=tk.W, pady=(0, 6))

        tree = ttk.Treeview(frame, columns=("label", "key"), show="headings", height=12)
        tree.heading("label", text="显示名称")
        tree.heading("key", text="内部名称 (目录/模型)")
        tree.column("label", width=200, anchor=tk.W)
        tree.column("key", width=250, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        self._label_tree = tree

        def reload_tree():
            tree.delete(*tree.get_children())
            for i, (key, label) in enumerate(zip(CLASSES, CLASS_LABELS)):
                tree.insert("", tk.END, iid=str(i), values=(label, key))

        reload_tree()

        btnrow = ttk.Frame(frame, style="Dark.TFrame")
        btnrow.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btnrow, text="添加", width=8,
                   command=lambda: self._add_label(reload_tree)).pack(side=tk.LEFT)
        ttk.Button(btnrow, text="重命名显示名", width=12,
                   command=lambda: self._rename_label(reload_tree)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btnrow, text="重命名内部名", width=12,
                   command=lambda: self._rename_class_key(reload_tree)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btnrow, text="删除", width=8,
                   command=lambda: self._remove_label(reload_tree)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btnrow, text="完成", width=8,
                   command=dlg.destroy).pack(side=tk.RIGHT)

    def _label_selection(self) -> int:
        sel = self._label_tree.selection()
        return int(sel[0]) if sel else -1

    def _apply_class_change(self, reload_tree):
        """标签变化后统一刷新界面并持久化"""
        _save_classes(CLASSES, CLASS_LABELS)
        self._rebuild_class_buttons()
        self._update_class_summary()
        self._update_stats()
        self._update_data_stats()
        self._update_dataset_stats()
        reload_tree()
        self.status_var.set(f"分类标签已更新: 共 {len(CLASSES)} 类")

    def _add_label(self, reload_tree):
        key = simpledialog.askstring("添加分类", "内部名称 (目录/模型类名，如 peanut):",
                                     parent=self.root)
        if not key:
            return
        key = self._validate_class_key(key)
        if not key:
            return
        label = simpledialog.askstring("添加分类", "显示名称 (界面文字，可中文):",
                                       parent=self.root) or ""
        label = label.strip() or key
        globals()["CLASSES"] = CLASSES + [key]
        globals()["CLASS_LABELS"] = CLASS_LABELS + [label]
        _add_class_dir(key)
        self._apply_class_change(reload_tree)

    def _rename_label(self, reload_tree):
        idx = self._label_selection()
        if idx < 0:
            messagebox.showwarning("提示", "请先选择一个分类")
            return
        label = simpledialog.askstring("重命名显示名",
                                       f"\"{CLASSES[idx]}\" 的新显示名称:",
                                       initialvalue=CLASS_LABELS[idx], parent=self.root)
        if not label:
            return
        label = label.strip() or CLASS_LABELS[idx]
        new_labels = list(CLASS_LABELS)
        new_labels[idx] = label
        globals()["CLASS_LABELS"] = new_labels
        self._apply_class_change(reload_tree)

    def _rename_class_key(self, reload_tree):
        idx = self._label_selection()
        if idx < 0:
            messagebox.showwarning("提示", "请先选择一个分类")
            return
        key = simpledialog.askstring("重命名内部名",
                                     f"\"{CLASS_LABELS[idx]}\" 的新内部名称 (目录/模型类名):",
                                     initialvalue=CLASSES[idx], parent=self.root)
        if not key:
            return
        key = self._validate_class_key(key, exclude=idx)
        if not key:
            return
        if key == CLASSES[idx]:
            return
        _rename_class_dir(CLASSES[idx], key)
        new_classes = list(CLASSES)
        new_classes[idx] = key
        globals()["CLASSES"] = new_classes
        self._apply_class_change(reload_tree)

    def _remove_label(self, reload_tree):
        idx = self._label_selection()
        if idx < 0:
            messagebox.showwarning("提示", "请先选择一个分类")
            return
        if len(CLASSES) <= 1:
            messagebox.showerror("错误", "至少保留一个分类")
            return
        n = sum(len(_list_images(os.path.join(_dataset_dir(ds), "raw", CLASSES[idx])))
                for ds in _list_datasets())
        if not messagebox.askyesno("删除分类",
                                   f"确定删除分类 \"{CLASS_LABELS[idx]}\"？\n"
                                   f"将删除其所有数据 ({n} 张图片) 及各数据集中的目录，不可恢复。",
                                   parent=self.root):
            return
        _remove_class_dir(CLASSES[idx])
        globals()["CLASSES"] = [k for i, k in enumerate(CLASSES) if i != idx]
        globals()["CLASS_LABELS"] = [l for i, l in enumerate(CLASS_LABELS) if i != idx]
        self._apply_class_change(reload_tree)

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
        ttk.Label(ctrl, text="选择类别 (按键 1-9)", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.class_btns_frame = ttk.Frame(ctrl, style="Dark.TFrame")
        self.class_btns_frame.pack(fill=tk.X)
        self._rebuild_class_buttons()

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

        ttk.Label(ctrl, text="\n提示: 按键 1-9 或点击按钮选择类别\n"
                             "每个类别对应数据集中的一个目录，\n类别可在第 0 步「分类标签」中增删改。",
                  style="Dark.TLabel", justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

    def _build_prepare_tab(self):
        tab = self.tab_prepare

        box = ttk.Frame(tab, style="Dark.TFrame")
        box.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        ttk.Label(box, text="数据集准备 (自动划分训练/验证集)",
                  style="Section.TLabel").pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(box, text="当前数据集目录:", style="Dark.TLabel").pack(anchor=tk.W)
        self.data_root_label = ttk.Label(box, text=self.dataset_dir, style="Dark.TLabel",
                                         foreground="#5a9bd5")
        self.data_root_label.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(box, text="原始数据统计 (当前数据集/raw):", style="Dark.TLabel").pack(anchor=tk.W)
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
                            "划分会把当前数据集 raw 下的图片按比例复制到 train 和 val。",
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

        ttk.Label(left, text="输出模型文件名 (不含后缀，自动生成 .pt 和 .onnx):", style="Dark.TLabel").pack(anchor=tk.W)
        self.out_name_var = tk.StringVar(value="smart_glasses_cls")
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
        if idx < 0 or idx >= len(CLASSES):
            return
        self.class_index = idx
        colors = ["#00d4aa", "#f39c12", "#9b59b6", "#e74c3c", "#3498db"]
        for i, btn in enumerate(self.class_buttons):
            if i == idx:
                btn.config(bg=colors[idx % len(colors)], fg="#1a1a2e")
            else:
                btn.config(bg="#16213e", fg="#e0e0e0")

    def _rebuild_class_buttons(self):
        for w in self.class_btns_frame.winfo_children():
            w.destroy()
        self.class_buttons = []
        for i, label in enumerate(CLASS_LABELS):
            btn = tk.Button(self.class_btns_frame, text=f"{label} ({i + 1})",
                            font=("Consolas", 11, "bold"),
                            relief="flat", cursor="hand2", height=1,
                            command=lambda idx=i: self._select_class(idx))
            btn.pack(fill=tk.X, pady=2)
            self.class_buttons.append(btn)
        if self.class_index >= len(CLASSES):
            self.class_index = 0
        self._select_class(self.class_index)

    def _save_frame(self):
        if self._current_frame is None:
            self.status_var.set("没有可保存的帧")
            return
        cls = CLASSES[self.class_index]
        d = os.path.join(self.dataset_dir, "raw", cls)
        os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{cls}_{ts}.jpg"
        path = os.path.join(d, name)
        frame_to_save = self._current_frame
        if self.offset_x or self.offset_y:
            frame_to_save = self._apply_offset(self._current_frame)
        # cv2.imwrite 在 Windows 上不支持中文路径（如中文数据集名），
        # 静默失败导致计数不增长；改用 imencode + 原生写入以支持 Unicode 路径
        ok, buf = cv2.imencode(".jpg", frame_to_save)
        if ok:
            try:
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
            except OSError as e:
                self.status_var.set(f"保存失败: {path} ({e})")
                return
            self.counts[self.class_index] += 1
            self.status_var.set(f"已保存: {name}")
            self._update_stats()
            self._update_data_stats()
            self._update_dataset_stats()
        else:
            self.status_var.set(f"保存失败: {path}")

    def _update_stats(self):
        self.counts = _count_by_class(os.path.join(self.dataset_dir, "raw"))
        lines = [f"{CLASS_LABELS[i]}: {self.counts[i]} 张" for i in range(len(CLASSES))]
        self.stats_var.set("\n".join(lines))

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
        cls_info = " ".join(f"{CLASS_LABELS[i]}:{self.counts[i]}" for i in range(len(CLASSES)))
        cv2.putText(disp, f"DS: {self.dataset_name} | Class: {CLASSES[self.class_index]} | {cls_info}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

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

        raw_root = os.path.join(self.dataset_dir, "raw")
        total_raw = sum(len(_list_images(os.path.join(raw_root, c))) for c in CLASSES)
        if total_raw == 0:
            messagebox.showerror("错误", "原始数据为空，请先在第 1 步采集数据")
            return

        self.status_var.set("正在划分数据集...")
        self.root.update_idletasks()

        try:
            _prepare_dataset(self.dataset_dir, val_ratio)
        except RuntimeError as e:
            messagebox.showerror("错误", str(e))
            return

        self._update_data_stats()
        self.status_var.set("数据集划分完成")
        messagebox.showinfo("完成", "数据集划分完成，可在第 3 步开始训练")

    def _update_data_stats(self):
        raw_counts = _count_by_class(os.path.join(self.dataset_dir, "raw"))
        self.raw_stats_var.set(
            " | ".join(f"{CLASS_LABELS[i]}: {raw_counts[i]} 张" for i in range(len(CLASSES)))
        )
        train_counts = _count_by_class(os.path.join(self.dataset_dir, "train"))
        val_counts = _count_by_class(os.path.join(self.dataset_dir, "val"))
        tr_line = " | ".join(f"{CLASS_LABELS[i]} {train_counts[i]}" for i in range(len(CLASSES)))
        va_line = " | ".join(f"{CLASS_LABELS[i]} {val_counts[i]}" for i in range(len(CLASSES)))
        self.prepared_stats_var.set(f"训练集: {tr_line}\n验证集: {va_line}")
        pairs = " | ".join(f"{CLASS_LABELS[i]} {train_counts[i]}/{val_counts[i]}"
                           for i in range(len(CLASSES)))
        self.train_stats_var.set(
            f"训练集 {sum(train_counts)} 张 | 验证集 {sum(val_counts)} 张\n({pairs})"
        )

    # ───────────── 训练 ─────────────

    def _base_model_name(self) -> str:
        """返回不含后缀的模型基础名（自动剥离 .pt/.onnx）"""
        name = self.out_name_var.get().strip() or "smart_glasses_cls"
        for ext in (".pt", ".onnx"):
            if name.lower().endswith(ext):
                name = name[: -len(ext)]
                break
        return name

    def _start_training(self):
        if self.train_proc and self.train_proc.is_alive():
            return

        train_root = os.path.join(self.dataset_dir, "train")
        total = sum(len(_list_images(os.path.join(train_root, c))) for c in CLASSES)
        if total == 0:
            messagebox.showerror("错误", "训练集为空，请先采集数据并划分数据集")
            return

        try:
            cfg = {
                "data_root": self.dataset_dir,
                "base_model": self.base_var.get().strip() or "yolo11n-cls.pt",
                "epochs": max(1, int(self.param_vars["epochs_var"].get())),
                "imgsz": max(32, int(self.param_vars["imgsz_var"].get())),
                "batch": max(1, int(self.param_vars["batch_var"].get())),
                "device": self.param_vars["device_var"].get().strip(),
                "workers": max(0, int(self.param_vars["workers_var"].get())),
                "patience": max(0, int(self.param_vars["patience_var"].get())),
                "run_name": f"smart_glasses_{self.dataset_name}",
                "out_dir": self.dataset_dir,
                "out_name": self._base_model_name(),
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
        self.status_var.set(f"训练进行中 (数据集: {self.dataset_name})...")
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
            self.status_var.set("训练已结束，模型位于数据集目录")

    def _drain_log(self):
        try:
            while True:
                chunk = self.log_q.get_nowait()
                self._append_log(chunk)
        except (queue.Empty, ValueError, OSError):
            pass

    def _append_log(self, chunk: str):
        chunk = _ANSI_RE.sub("", chunk)
        if not chunk:
            return
        self.log_text.config(state="normal")
        # 处理 tqdm 的 \r 进度：覆盖最后一行
        parts = chunk.split("\r")
        for i, part in enumerate(parts):
            if not part:
                continue
            if i < len(parts) - 1 or chunk.endswith("\r"):
                if "\n" in part:
                    for ln in part.split("\n"):
                        if ln and not _is_noise_line(ln):
                            self._replace_last_line(ln)
                else:
                    if not _is_noise_line(part):
                        self._replace_last_line(part)
            else:
                kept = [ln for ln in part.split("\n") if not _is_noise_line(ln)]
                if kept:
                    self.log_text.insert(tk.END, "\n".join(kept) +
                                         ("\n" if part.endswith("\n") else ""))
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

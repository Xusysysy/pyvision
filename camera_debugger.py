"""
摄像头调试工具 v2 — camera_debugger.py
用法: python camera_debugger.py [--camera 0] [--width 640] [--height 480]

依赖:
  基础: python -m pip install opencv-python Pillow numpy
  CNN:  python -m pip install ultralytics   (YOLO .pt 模型)
        python -m pip install onnxruntime    (ONNX 模型)
"""

import cv2
import numpy as np
import time
import os
import sys
import argparse
import threading
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    raise SystemExit("需要 tkinter (Linux 下: sudo apt install python3-tk)")

# 在 Windows 上启用 DPI 感知以减少 Tk 界面的模糊（必须在创建 Tk 实例之前）
if sys.platform == "win32":
    try:
        import ctypes
        # 优先使用 Per-Monitor DPI awareness（如果可用），否则回退到 SetProcessDPIAware
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception:
        pass

# PyInstaller 打包后修复 SSL 证书路径
if getattr(sys, "frozen", False):
    import ssl
    import certifi
    ssl._create_default_verify_paths = lambda: None
    ssl._create_default_https_context = ssl._create_unverified_context

# ═══════════════════════════════════════════════
# 1. 图像处理接口
# ═══════════════════════════════════════════════

class FrameProcessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        pass


class NoOpProcessor(FrameProcessor):
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


class EdgeDetectionProcessor(FrameProcessor):
    def __init__(self, low=50, high=150):
        self.low = low
        self.high = high

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.low, self.high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


class GrayscaleProcessor(FrameProcessor):
    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class CNNProcessor(FrameProcessor):
    """
    通用 CNN 视觉识别处理器
    自动检测模型格式并加载:
      - .pt / .yaml → ultralytics YOLO
      - .onnx       → onnxruntime
    提供三种图像输入接口供子类或外部模型推理:
      - frame  : 原始 BGR 图像
      - gray   : 灰度三通道图像
      - edges  : Canny 边缘检测三通道图像
    """

    DEFAULT_MODEL = "yolov8n.pt"

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        if os.path.isfile(path):
            return path
        if getattr(sys, "frozen", False):
            base = sys._MEIPASS
            bundled = os.path.join(base, os.path.basename(path))
            if os.path.isfile(bundled):
                return bundled
        return path

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.model_path = self._resolve_model_path(model_path)
        self.conf_threshold = conf_threshold
        self.model = None
        self._model_type = None  # "yolo_ultralytics" | "onnx"
        self._onnx_session = None
        self._onnx_input_name = None
        self._yolo_names = {}
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.isfile(model_path):
            print(f"[CNN] 模型文件不存在: {model_path}，将在首次推理时提示")
            return

        ext = os.path.splitext(model_path)[1].lower()

        if ext in (".pt", ".yaml"):
            self._load_yolo_ultralytics(model_path)
        elif ext == ".onnx":
            self._load_onnx(model_path)
        else:
            print(f"[CNN] 不支持的模型格式: {ext}，尝试作为 ONNX 加载")
            self._load_onnx(model_path)

    def _load_yolo_ultralytics(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self._model_type = "yolo_ultralytics"
            self._yolo_names = self.model.names if hasattr(self.model, "names") else {}
            print(f"[CNN] 已加载 YOLO 模型: {model_path}")
        except ImportError:
            print("[CNN] 未安装 ultralytics，无法加载 .pt 模型。pip install ultralytics")
        except Exception as e:
            print(f"[CNN] YOLO 模型加载失败: {e}")

    def _load_onnx(self, model_path: str):
        try:
            import onnxruntime as ort
            self._onnx_session = ort.InferenceSession(model_path)
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            self.model = self._onnx_session
            self._model_type = "onnx"
            print(f"[CNN] 已加载 ONNX 模型: {model_path}")
        except ImportError:
            print("[CNN] 未安装 onnxruntime，无法加载模型。pip install onnxruntime")
        except Exception as e:
            print(f"[CNN] ONNX 模型加载失败: {e}")

    # ──────── 图像转换接口 ────────

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _to_edges(frame: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # ──────── 推理 ────────

    def detect(self, frame: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> list[dict]:
        """
        在三种输入上执行检测，返回检测结果列表。
        每个结果为 dict，至少包含:
          {"bbox": [x, y, w, h], "label": str, "confidence": float}
        子类可重写此方法实现自定义推理。
        """
        if self._model_type == "yolo_ultralytics":
            return self._detect_yolo(frame)
        elif self._model_type == "onnx":
            return self._detect_onnx(frame)
        return []

    def _detect_yolo(self, frame: np.ndarray) -> list[dict]:
        results = []
        try:
            preds = self.model(frame, conf=self.conf_threshold, verbose=False)
            for pred in preds:
                boxes = pred.boxes
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    label = self._yolo_names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = xyxy
                    results.append({
                        "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                        "label": label,
                        "confidence": conf,
                    })
        except Exception as e:
            print(f"[CNN] YOLO 推理失败: {e}")
        return results

    def _detect_onnx(self, frame: np.ndarray) -> list[dict]:
        if self._onnx_session is None:
            return []
        results = []
        try:
            input_meta = self._onnx_session.get_inputs()[0]
            _, _, req_h, req_w = input_meta.shape
            resized = cv2.resize(frame, (req_w, req_h))
            blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
            if blob.max() > 1.0:
                blob /= 255.0

            output = self._onnx_session.run(None, {self._onnx_input_name: blob})[0]
            output = output[0]

            if output.ndim == 2 and output.shape[1] == 6:
                for det in output:
                    x1, y1, x2, y2, conf, cls_id = det[:6]
                    if conf < self.conf_threshold:
                        continue
                    sx, sy = frame.shape[1] / req_w, frame.shape[0] / req_h
                    x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                    results.append({
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "label": str(int(cls_id)),
                        "confidence": float(conf),
                    })
            elif output.ndim == 3:
                for det in output:
                    if det.shape[0] >= 5:
                        conf = float(det[4])
                        if conf < self.conf_threshold:
                            continue
                        cx, cy, w, h = det[:4]
                        sx, sy = frame.shape[1] / req_w, frame.shape[0] / req_h
                        x1 = int((cx - w / 2) * sx)
                        y1 = int((cy - h / 2) * sy)
                        results.append({
                            "bbox": [x1, y1, int(w * sx), int(h * sy)],
                            "label": str(int(det[5])) if det.shape[0] > 5 else "0",
                            "confidence": conf,
                        })
        except Exception as e:
            print(f"[CNN] ONNX 推理失败: {e}")
        return results

    # ──────── 绘制结果 ────────

    def _draw_results(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        display = frame.copy()
        for r in results:
            x, y, w, h = r["bbox"]
            label = r.get("label", "unknown")
            conf = r.get("confidence", 0.0)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(display, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return display

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(frame)
        edges = self._to_edges(frame)

        if self.model is None:
            cv2.putText(frame, "CNN: model not loaded", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        try:
            results = self.detect(frame, gray, edges)
            return self._draw_results(frame, results)
        except Exception as e:
            cv2.putText(frame, f"CNN error: {e}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame


# ═══════════════════════════════════════════════
# 2. 摄像头枚举
# ═══════════════════════════════════════════════

def enumerate_cameras(max_id: int = 10) -> list[dict]:
    """
    扫描系统可用摄像头，返回列表:
    [{"id": 0, "name": "摄像头 0", "width": 640, "height": 480}, ...]
    """
    found = []
    for i in range(max_id):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 尝试读取后端名称
            backend = cap.getBackendName() if hasattr(cap, 'getBackendName') else ""
            name = f"摄像头 {i}"
            if backend:
                name += f" ({backend})"
            found.append({"id": i, "name": name, "width": w, "height": h})
            cap.release()
    return found


# ═══════════════════════════════════════════════
# 3. 摄像头管理
# ═══════════════════════════════════════════════

class CameraManager:
    """管理摄像头并在后台线程中持续采集最新一帧，read() 快速返回缓存帧。

    这样可以避免在 UI 线程中直接调用 blocking 的 cap.read()，减少界面拖动或交互时的卡顿。
    """
    def __init__(self, camera_id: int = 0, width: int = 1080, height: int = 720):
        self.camera_id = camera_id
        self.requested_width = width
        self.requested_height = height
        self.cap = None
        self.lock = threading.Lock()        # 用于保护 cap 的打开/关闭/切换
        self._frame_lock = threading.Lock() # 用于保护 _last_frame
        self._last_frame = None
        self._last_ret = False
        self._running = True
        self._grab_thread = None

        # 尝试打开摄像头
        self._open()

        # 启动后台采集线程
        self._start_grab_thread()

    def _open(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")
        # 尝试设置请求的分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] 已打开摄像头 {self.camera_id}: {self.actual_width}x{self.actual_height}")

    def _start_grab_thread(self):
        if self._grab_thread and self._grab_thread.is_alive():
            return
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def _grab_loop(self):
        while getattr(self, '_running', False):
            # 把对 cap 的访问放在 lock 里，保证与切换/释放互斥
            with self.lock:
                cap = self.cap
                if cap and cap.isOpened():
                    try:
                        ret, frame = cap.read()
                    except Exception:
                        ret, frame = False, None
                else:
                    ret, frame = False, None

            with self._frame_lock:
                self._last_ret = bool(ret)
                if ret and frame is not None:
                    # 仅保存副本以避免被外部修改
                    self._last_frame = frame.copy()
                else:
                    self._last_frame = None

            time.sleep(0.01)

    def read(self) -> tuple[bool, np.ndarray | None]:
        # 非阻塞地返回后台缓存的最新一帧
        with self._frame_lock:
            if self._last_frame is None:
                return self._last_ret, None
            return self._last_ret, self._last_frame.copy()

    def switch(self, new_id: int):
        """切换摄像头（线程安全）"""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            self.camera_id = new_id

        # 重新打开（不在 lock 中，以免阻塞采集线程长时间等待）
        self._open()

    def set_resolution(self, width: int, height: int) -> bool:
        """设置分辨率。优先尝试直接设置属性，若后端不支持则重启摄像头。"""
        # 尝试快速路径：直接设置属性
        with self.lock:
            if self.cap and self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_w == width and actual_h == height:
                        self.actual_width = actual_w
                        self.actual_height = actual_h
                        print(f"[Camera] 分辨率已改为: {self.actual_width}x{self.actual_height}")
                        return True
                except Exception:
                    pass

                # 如果直接设置不生效，释放摄像头并在外部重新打开
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        # 更新请求分辨率并重试打开（_open 会使用 self.requested_*）
        self.requested_width = width
        self.requested_height = height
        try:
            self._open()
            return True
        except Exception as e:
            print(f"[Camera] 无法设置分辨率: {e}")
            return False

    def release(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                print("[Camera] 已释放摄像头")

    def stop(self):
        """停止后台采集线程并释放资源"""
        self._running = False
        if self._grab_thread:
            self._grab_thread.join(timeout=1.0)
        self.release()
        print("[Camera] 已停止采集线程")

    def get_info(self) -> dict:
        if self.cap and self.cap.isOpened():
            return {
                "id": self.camera_id,
                "width": self.actual_width,
                "height": self.actual_height,
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
            }
        return {}


# ═══════════════════════════════════════════════
# 4. GUI 主界面
# ═══════════════════════════════════════════════

class CameraDebuggerGUI:

    PROCESSORS = {
        "直通 (原始)": NoOpProcessor,
        "灰度": GrayscaleProcessor,
        "边缘检测 (Canny)": EdgeDetectionProcessor,
        "CNN 模型": CNNProcessor,
    }

    def __init__(self, root: tk.Tk, camera: CameraManager):
        self.root = root
        self.camera = camera
        self.processor: FrameProcessor = NoOpProcessor()
        self.running = True
        self.show_fps = True
        self.show_crosshair = False
        self.mirror = False
        self.recording = False
        self.video_writer = None

        # 默认输出目录
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "snapshots"))
        os.makedirs(self.output_dir.get(), exist_ok=True)

        # 照片命名前缀
        self.photo_prefix = tk.StringVar(value="snap")

        # 照片计数器
        self._photo_count = 0

        # 分辨率设置
        self.resolution_presets = [
            (640, 480),
            (800, 600),
            (1024, 768),
            (1280, 720),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
        ]
        self.custom_width = tk.StringVar(value=str(camera.actual_width))
        self.custom_height = tk.StringVar(value=str(camera.actual_height))

        # FPS
        self._frame_count = 0
        self._fps_time = time.time()
        self._current_fps = 0.0

        # 帧缓存
        self._current_frame = None
        self._processed_frame = None

        # 摄像头列表
        self._camera_list: list[dict] = []

        self._setup_ui()
        self._scan_cameras()
        self._update_frame()

    # ───────────── UI 构建 ─────────────

    def _setup_ui(self):
        self.root.title("Camera Debugger v2")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1000, 680)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#1a1a2e")
        style.configure("Dark.TLabel", background="#1a1a2e", foreground="#e0e0e0",
                         font=("Consolas", 10))
        style.configure("Title.TLabel", background="#1a1a2e", foreground="#00d4aa",
                         font=("Consolas", 14, "bold"))
        style.configure("Section.TLabel", background="#1a1a2e", foreground="#5a9bd5",
                         font=("Consolas", 10, "bold"))
        style.configure("Status.TLabel", background="#16213e", foreground="#a0a0a0",
                         font=("Consolas", 9))
        style.configure("Dark.TButton", font=("Consolas", 10))
        style.configure("Accent.TButton", font=("Consolas", 10, "bold"))
        style.configure("Dark.TCheckbutton", background="#1a1a2e", foreground="#e0e0e0",
                         font=("Consolas", 10))
        style.configure("Dark.TEntry", fieldbackground="#16213e", foreground="#e0e0e0",
                         font=("Consolas", 9))

        # 主布局
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 顶部标题
        header = ttk.Frame(main, style="Dark.TFrame")
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="Camera Debugger v2", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="  |  调试 + 拍照 + 切换摄像头", style="Dark.TLabel").pack(side=tk.LEFT)

        # 中间主体
        body = ttk.Frame(main, style="Dark.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        # ─── 视频区域 ───
        video_frame = ttk.Frame(body, style="Dark.TFrame")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(video_frame, bg="#0d0d0d", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ─── 右侧控制面板 ───
        ctrl = ttk.Frame(body, style="Dark.TFrame", width=280)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        ctrl.pack_propagate(False)

        # 用 Canvas + Scrollbar 实现可滚动面板
        ctrl_canvas = tk.Canvas(ctrl, bg="#1a1a2e", highlightthickness=0)
        ctrl_scrollbar = ttk.Scrollbar(ctrl, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        ctrl_inner = ttk.Frame(ctrl_canvas, style="Dark.TFrame")

        ctrl_inner.bind("<Configure>",
                        lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.create_window((0, 0), window=ctrl_inner, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)

        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctrl_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        panel = ctrl_inner  # 后续控件都放这里

        # ══════ 摄像头选择 ══════
        ttk.Label(panel, text="摄像头", style="Section.TLabel").pack(anchor=tk.W, pady=(4, 4))

        cam_row = ttk.Frame(panel, style="Dark.TFrame")
        cam_row.pack(fill=tk.X, pady=(0, 4))

        self.camera_var = tk.StringVar(value="摄像头 0")
        self.camera_combo = ttk.Combobox(cam_row, textvariable=self.camera_var,
                                          state="readonly", width=18)
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_switch)

        ttk.Button(cam_row, text="刷新", width=5,
                   command=self._scan_cameras).pack(side=tk.LEFT, padx=(4, 0))

        self.cam_info_label = ttk.Label(panel, text="分辨率: --", style="Dark.TLabel")
        self.cam_info_label.pack(anchor=tk.W, pady=(0, 8))

        # ══════ 分辨率设置 ══════
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text="分辨率设置", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))

        # 预设分辨率
        ttk.Label(panel, text="预设:", style="Dark.TLabel").pack(anchor=tk.W)
        preset_row = ttk.Frame(panel, style="Dark.TFrame")
        preset_row.pack(fill=tk.X, pady=(2, 4))

        preset_options = [f"{w}x{h}" for w, h in self.resolution_presets]
        self.preset_var = tk.StringVar(value=preset_options[0])
        preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var,
                                     values=preset_options, state="readonly", width=12)
        preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(preset_row, text="应用", width=6,
                   command=self._apply_preset_resolution).pack(side=tk.LEFT, padx=(4, 0))

        # 自定义分辨率
        ttk.Label(panel, text="自定义:", style="Dark.TLabel").pack(anchor=tk.W, pady=(4, 2))
        custom_row = ttk.Frame(panel, style="Dark.TFrame")
        custom_row.pack(fill=tk.X, pady=(2, 4))

        ttk.Label(custom_row, text="W:", style="Dark.TLabel").pack(side=tk.LEFT)
        ttk.Entry(custom_row, textvariable=self.custom_width, width=6).pack(side=tk.LEFT, padx=(2, 4))

        ttk.Label(custom_row, text="H:", style="Dark.TLabel").pack(side=tk.LEFT)
        ttk.Entry(custom_row, textvariable=self.custom_height, width=6).pack(side=tk.LEFT, padx=(2, 4))

        ttk.Button(custom_row, text="应用", width=6,
                   command=self._apply_custom_resolution).pack(side=tk.LEFT)

        # ══════ 处理管线 ══════
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text="处理管线", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))

        self.processor_var = tk.StringVar(value="直通 (原始)")
        proc_menu = ttk.Combobox(panel, textvariable=self.processor_var,
                                  values=list(self.PROCESSORS.keys()),
                                  state="readonly", width=24)
        proc_menu.pack(fill=tk.X, pady=(0, 8))
        proc_menu.bind("<<ComboboxSelected>>", self._on_processor_change)

        # ══════ 显示选项 ══════
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text="显示选项", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))

        self.fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="显示 FPS", variable=self.fps_var,
                        style="Dark.TCheckbutton",
                        command=self._toggle_fps).pack(anchor=tk.W)

        self.crosshair_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="十字准星", variable=self.crosshair_var,
                        style="Dark.TCheckbutton",
                        command=self._toggle_crosshair).pack(anchor=tk.W)

        self.mirror_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="水平镜像", variable=self.mirror_var,
                        style="Dark.TCheckbutton",
                        command=self._toggle_mirror).pack(anchor=tk.W)

        # ══════ 拍照设置 ══════
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text="拍照设置", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))

        # 输出目录
        ttk.Label(panel, text="输出目录:", style="Dark.TLabel").pack(anchor=tk.W)
        dir_row = ttk.Frame(panel, style="Dark.TFrame")
        dir_row.pack(fill=tk.X, pady=(2, 4))

        self.dir_entry = ttk.Entry(dir_row, textvariable=self.output_dir, width=20)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_row, text="浏览...", width=6,
                   command=self._choose_output_dir).pack(side=tk.LEFT, padx=(4, 0))

        # 文件名前缀
        ttk.Label(panel, text="文件名前缀:", style="Dark.TLabel").pack(anchor=tk.W)
        ttk.Entry(panel, textvariable=self.photo_prefix, width=24).pack(fill=tk.X, pady=(2, 4))

        # 计数器显示
        self.count_label = ttk.Label(panel, text="已拍: 0 张", style="Dark.TLabel")
        self.count_label.pack(anchor=tk.W, pady=(0, 4))

        # ══════ 操作按钮 ══════
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text="操作", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))

        btn_grid = ttk.Frame(panel, style="Dark.TFrame")
        btn_grid.pack(fill=tk.X, pady=(0, 4))

        # 拍照按钮 — 大按钮
        self.snap_btn = tk.Button(
            btn_grid, text="拍照 (S)", font=("Consolas", 12, "bold"),
            bg="#00d4aa", fg="#1a1a2e", activebackground="#00b894",
            relief="flat", cursor="hand2", height=2,
            command=self._snapshot
        )
        self.snap_btn.pack(fill=tk.X, pady=(0, 6))

        # 录制 + 打开目录
        self.record_btn_text = tk.StringVar(value="开始录制 (R)")
        ttk.Button(btn_grid, textvariable=self.record_btn_text,
                   style="Dark.TButton",
                   command=self._toggle_recording).pack(fill=tk.X, pady=2)

        ttk.Button(btn_grid, text="打开输出目录",
                   style="Dark.TButton",
                   command=self._open_output_dir).pack(fill=tk.X, pady=2)

        # ══════ 摄像头信息 ══════
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text="状态信息", style="Section.TLabel").pack(anchor=tk.W, pady=(6, 4))

        self.info_label = ttk.Label(panel, text="--", style="Dark.TLabel", justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)

        self.mouse_label = ttk.Label(panel, text="鼠标: (-, -)", style="Dark.TLabel")
        self.mouse_label.pack(anchor=tk.W, pady=(8, 0))

        self.pixel_label = ttk.Label(panel, text="像素: --", style="Dark.TLabel")
        self.pixel_label.pack(anchor=tk.W)

        # ─── 鼠标事件 ───
        self.canvas.bind("<Motion>", self._on_mouse_move)

        # ─── 底部状态栏 ───
        status_frame = ttk.Frame(main, style="Dark.TFrame")
        status_frame.pack(fill=tk.X, pady=(8, 0))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var,
                  style="Status.TLabel").pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.frame_info_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.frame_info_var,
                  style="Status.TLabel").pack(side=tk.RIGHT)

        # ─── 快捷键 ───
        self.root.bind("<s>", lambda e: self._snapshot())
        self.root.bind("<S>", lambda e: self._snapshot())
        self.root.bind("<r>", lambda e: self._toggle_recording())
        self.root.bind("<R>", lambda e: self._toggle_recording())
        self.root.bind("<q>", lambda e: self._on_close())
        self.root.bind("<Q>", lambda e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ───────────── 摄像头扫描与切换 ─────────────

    def _scan_cameras(self):
        """扫描系统可用摄像头并更新下拉列表"""
        self.status_var.set("正在扫描摄像头...")
        self.root.update_idletasks()

        # 在子线程中扫描避免 UI 冻结
        def _do_scan():
            cams = enumerate_cameras(max_id=6)
            self.root.after(0, lambda: self._on_scan_done(cams))

        threading.Thread(target=_do_scan, daemon=True).start()

    def _on_scan_done(self, cams: list[dict]):
        self._camera_list = cams
        names = [c["name"] for c in cams]

        if not names:
            names = ["未检测到摄像头"]
            self.status_var.set("未检测到摄像头")
        else:
            self.status_var.set(f"检测到 {len(cams)} 个摄像头")

        self.camera_combo["values"] = names

        # 如果当前摄像头还在列表中就保持，否则选第一个
        current_id = self.camera.camera_id
        idx = next((i for i, c in enumerate(cams) if c["id"] == current_id), 0)
        if names:
            self.camera_combo.current(idx)

        self._update_cam_info()

    def _on_camera_switch(self, event=None):
        """切换摄像头"""
        sel = self.camera_combo.current()
        if sel < 0 or sel >= len(self._camera_list):
            return

        new_id = self._camera_list[sel]["id"]
        if new_id == self.camera.camera_id:
            return

        self.status_var.set(f"切换到摄像头 {new_id}...")
        self.root.update_idletasks()

        try:
            self.camera.switch(new_id)
            self.status_var.set(f"已切换到 {self._camera_list[sel]['name']}")
            self._update_cam_info()
        except RuntimeError as e:
            messagebox.showerror("切换失败", str(e))
            self.status_var.set("切换失败")

    def _update_cam_info(self):
        info = self.camera.get_info()
        if info:
            fps = info.get('fps', 0)
            fps_str = f"{fps:.0f}" if fps else "--"
            self.cam_info_label.config(
                text=f"分辨率: {info['width']}x{info['height']} | FPS: {fps_str}"
            )
            # 更新自定义输入框
            self.custom_width.set(str(info['width']))
            self.custom_height.set(str(info['height']))
        else:
            self.cam_info_label.config(text="分辨率: --")

    def _apply_preset_resolution(self):
        """应用预设分辨率"""
        preset_str = self.preset_var.get()
        if not preset_str:
            return
        try:
            w, h = map(int, preset_str.split('x'))
            self._apply_resolution(w, h)
        except (ValueError, AttributeError):
            messagebox.showerror("错误", "分辨率格式不正确")

    def _apply_custom_resolution(self):
        """应用自定义分辨率"""
        try:
            w = int(self.custom_width.get())
            h = int(self.custom_height.get())
            if w <= 0 or h <= 0:
                raise ValueError("分辨率必须大于 0")
            self._apply_resolution(w, h)
        except ValueError as e:
            messagebox.showerror("错误", f"分辨率输入错误: {e}")

    def _apply_resolution(self, width: int, height: int):
        """应用分辨率到摄像头"""
        self.status_var.set(f"正在设置分辨率 {width}x{height}...")
        self.root.update_idletasks()

        if self.camera.set_resolution(width, height):
            self.status_var.set(f"分辨率已改为 {width}x{height}")
            self._update_cam_info()
        else:
            messagebox.showerror("失败", "无法设置分辨率")
            self.status_var.set("设置分辨率失败")

    # ───────────── 处理器切换 ─────────────

    def _on_processor_change(self, event=None):
        name = self.processor_var.get()
        cls = self.PROCESSORS.get(name, NoOpProcessor)
        try:
            if cls == CNNProcessor:
                path = CNNProcessor.DEFAULT_MODEL
                if not os.path.isfile(path):
                    path = filedialog.askopenfilename(
                        title="选择模型文件",
                        filetypes=[
                            ("YOLO 模型", "*.pt *.yaml"),
                            ("ONNX 模型", "*.onnx"),
                            ("所有文件", "*.*"),
                        ],
                    )
                if not path:
                    self.processor_var.set("直通 (原始)")
                    self.processor = NoOpProcessor()
                    return
                self.processor = cls(model_path=path)
            else:
                self.processor = cls()
            self.status_var.set(f"处理管线: {name}")
        except Exception as ex:
            messagebox.showerror("错误", f"加载处理器失败:\n{ex}")
            self.processor = NoOpProcessor()

    # ───────────── 视频循环 ─────────────

    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.camera.read()
        if not ret or frame is None:
            self.root.after(30, self._update_frame)
            return

        # 保护当前帧和处理帧的访问，避免在分辨率切换或释放过程中出现 None 或竞争
        try:
            self._current_frame = frame.copy()
        except Exception:
            self._current_frame = None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        try:
            processed = self.processor.process(frame)
        except Exception as ex:
            processed = frame
            cv2.putText(processed, f"Error: {ex}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self._processed_frame = processed.copy()

        display = processed.copy()

        # FPS
        if self.show_fps:
            cv2.putText(display, f"FPS: {self._current_fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 十字准星
        if self.show_crosshair:
            h, w = display.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(display, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
            cv2.line(display, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

        # 录制红点
        if self.recording:
            cv2.circle(display, (display.shape[1] - 20, 20), 8, (0, 0, 255), -1)
            if self.video_writer:
                self.video_writer.write(processed)

        # 帧信息
        h, w = display.shape[:2]
        ch = display.shape[2] if len(display.shape) > 2 else 1
        self.frame_info_var.set(f"{w}x{h} | {ch}ch")

        self._display_frame(display)

        # FPS 计算
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = now

        self.root.after(1, self._update_frame)

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        img_w, img_h = img.size
        scale = min(cw / img_w, ch / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        self._tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._tk_image)

        self._scale = scale
        self._offset_x = (cw - new_w) // 2
        self._offset_y = (ch - new_h) // 2

    # ───────────── 鼠标 ─────────────

    def _on_mouse_move(self, event):
        if not hasattr(self, '_scale') or self._current_frame is None:
            return
        ox = (event.x - self._offset_x) / self._scale
        oy = (event.y - self._offset_y) / self._scale
        h, w = self._current_frame.shape[:2]

        if 0 <= ox < w and 0 <= oy < h:
            ix, iy = int(ox), int(oy)
            if self.mirror:
                ix = w - 1 - ix
            b, g, r = self._current_frame[iy, ix]
            self.mouse_label.config(text=f"坐标: ({ix}, {iy})")
            self.pixel_label.config(text=f"BGR: ({b},{g},{r})  #{r:02x}{g:02x}{b:02x}")
        else:
            self.mouse_label.config(text="坐标: (-, -)")
            self.pixel_label.config(text="像素: --")

    # ───────────── 输出目录 ─────────────

    def _choose_output_dir(self):
        chosen = filedialog.askdirectory(
            title="选择照片输出目录",
            initialdir=self.output_dir.get()
        )
        if chosen:
            self.output_dir.set(chosen)
            os.makedirs(chosen, exist_ok=True)
            self.status_var.set(f"输出目录: {chosen}")

    # ───────────── 拍照 ─────────────

    def _snapshot(self):
        if self._processed_frame is None and self._current_frame is None:
            self.status_var.set("没有可保存的帧")
            return

        out_dir = self.output_dir.get()
        os.makedirs(out_dir, exist_ok=True)

        prefix = self.photo_prefix.get().strip() or "snap"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{ts}.png"
        path = os.path.join(out_dir, filename)

        # 如果处理器是直通，优先保存原始帧（更高清）；否则保存处理后的帧
        if isinstance(self.processor, NoOpProcessor):
            if self._current_frame is None:
                self.status_var.set("没有可保存的帧")
                return
            frame_to_save = self._current_frame if not self.mirror else cv2.flip(self._current_frame, 1)
        else:
            if self._processed_frame is None:
                self.status_var.set("没有可保存的帧")
                return
            frame_to_save = self._processed_frame

        ok = cv2.imwrite(path, frame_to_save)
        if ok:
            self._photo_count += 1
            self.count_label.config(text=f"已拍: {self._photo_count} 张")
            self.status_var.set(f"已保存: {filename}")
        else:
            self.status_var.set(f"保存失败: {path}")

    # ───────────── 录制 ─────────────

    def _toggle_recording(self):
        if not self.recording:
            if self._processed_frame is None:
                return
            out_dir = self.output_dir.get()
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(out_dir, f"rec_{ts}.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            h, w = self._processed_frame.shape[:2]
            self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
            self.recording = True
            self.record_btn_text.set("停止录制 (R)")
            self.status_var.set(f"录制中: {os.path.basename(path)}")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn_text.set("开始录制 (R)")
            self.status_var.set("录制已停止")

    # ───────────── 其他 ─────────────

    def _open_output_dir(self):
        d = self.output_dir.get()
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(d)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", d])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", d])

    def _toggle_fps(self):
        self.show_fps = self.fps_var.get()

    def _toggle_crosshair(self):
        self.show_crosshair = self.crosshair_var.get()

    def _toggle_mirror(self):
        self.mirror = self.mirror_var.get()

    def _on_close(self):
        self.running = False
        if self.recording and self.video_writer:
            self.video_writer.release()
        self.camera.release()
        self.root.destroy()


# ═══════════════════════════════════════════════
# 5. 启动入口
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="摄像头调试工具 v2")
    parser.add_argument("--camera", type=int, default=0, help="初始摄像头 ID (默认 0)")
    parser.add_argument("--width", type=int, default=640, help="请求宽度")
    parser.add_argument("--height", type=int, default=480, help="请求高度")
    parser.add_argument("--output", type=str, default=None, help="截图输出目录")
    args = parser.parse_args()

    try:
        camera = CameraManager(args.camera, args.width, args.height)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    root = tk.Tk()
    root.geometry("1040x720")
    app = CameraDebuggerGUI(root, camera)

    # 命令行指定输出目录
    if args.output:
        app.output_dir.set(args.output)
        os.makedirs(args.output, exist_ok=True)

    try:
        root.mainloop()
    finally:
        # 确保摄像头后台线程关闭
        try:
            camera.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
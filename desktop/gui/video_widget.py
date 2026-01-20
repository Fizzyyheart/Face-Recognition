"""
视频预览组件 - 多线程优化版
线程架构:
1. UI线程: 界面渲染
2. 摄像头线程: 读取帧
3. 推理线程: 人脸检测和识别
"""

import time
from collections import deque
from pathlib import Path
from threading import Thread, Event
from typing import Optional, List, Dict

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from backend.app.db.database import SessionLocal
from backend.app.services.attendance_service import AttendanceService
from backend.app.face.detector import FaceRecognizer

# PIL 用于中文绘制
from PIL import Image, ImageDraw, ImageFont

# 中文字体
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
try:
    CHINESE_FONT = ImageFont.truetype(FONT_PATH, 20)
except Exception:
    CHINESE_FONT = None


class FrameBuffer:
    """线程安全的帧缓冲区"""

    def __init__(self, maxsize: int = 2):
        self.maxsize = maxsize
        self._frames: deque = deque(maxlen=maxsize)
        self._results: Dict = {}
        self._latest_frame = None
        self._latest_results = []

    def put_frame(self, frame_id: int, frame: np.ndarray):
        """放入新帧"""
        self._frames.append((frame_id, frame))
        self._latest_frame = frame

    def get_frame_for_inference(self) -> Optional[tuple]:
        """获取待推理的帧"""
        try:
            return self._frames.popleft()
        except IndexError:
            return None

    def put_results(self, frame_id: int, results: List):
        """存入推理结果"""
        self._results[frame_id] = results
        self._latest_results = results
        old_ids = [k for k in self._results if k < frame_id - 10]
        for k in old_ids:
            del self._results[k]

    def get_latest(self) -> tuple:
        """获取最新帧和结果"""
        return self._latest_frame, self._latest_results


class InferenceWorker(Thread):
    """推理工作线程"""

    def __init__(self, buffer: FrameBuffer, stop_event: Event):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.stop_event = stop_event
        self.recognizer: Optional[FaceRecognizer] = None
        self.attendance_service: Optional[AttendanceService] = None
        self.ready = Event()
        self.error: Optional[str] = None

    def run(self):
        """线程主循环"""
        try:
            self.recognizer = FaceRecognizer()
            db = SessionLocal()
            self.attendance_service = AttendanceService(db)
            self.attendance_service.recognizer = self.recognizer
            self.ready.set()
        except Exception as e:
            self.error = str(e)
            self.ready.set()
            return

        while not self.stop_event.is_set():
            item = self.buffer.get_frame_for_inference()
            if item is None:
                time.sleep(0.01)
                continue

            frame_id, frame = item
            try:
                results = self.attendance_service.process_frame(frame)
                self.buffer.put_results(frame_id, results)
            except Exception as e:
                print(f"推理错误: {e}")


class CameraWorker(Thread):
    """摄像头读取线程"""

    def __init__(
        self,
        camera_id: int,
        resolution: str,
        buffer: FrameBuffer,
        stop_event: Event,
    ):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.resolution = resolution
        self.buffer = buffer
        self.stop_event = stop_event
        self.fps = 0.0
        self.error: Optional[str] = None
        self.ready = Event()

    def run(self):
        """线程主循环"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.error = f"无法打开摄像头 {self.camera_id}"
            self.ready.set()
            return

        if self.resolution == "720p":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ready.set()

        frame_id = 0
        frame_count = 0
        fps_start = time.time()

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            frame_id += 1
            frame_count += 1

            if frame_id % 2 == 0:
                self.buffer.put_frame(frame_id, frame.copy())
            else:
                self.buffer._latest_frame = frame

            elapsed = time.time() - fps_start
            if elapsed > 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

        cap.release()


class VideoWidget(QWidget):
    """视频预览组件 - 多线程优化版"""

    checkin_signal = pyqtSignal(str, bool)
    status_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

        self.buffer: Optional[FrameBuffer] = None
        self.camera_worker: Optional[CameraWorker] = None
        self.inference_worker: Optional[InferenceWorker] = None
        self.stop_event: Optional[Event] = None

        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self._update_display)

        self.last_results = []
        self._last_frame = None
        self._emitted_checkins: set = set()  # 已发送签到信号的person_id集合

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border-radius: 8px; }"
        )
        self.video_label.setText("点击「开始识别」启动摄像头")

        layout.addWidget(self.video_label)

    def start_camera(self, camera_id: int = 1, resolution: str = "480p"):
        """启动摄像头（异步）"""
        if self.camera_worker and self.camera_worker.is_alive():
            return

        self.video_label.setText("正在初始化...")

        self.buffer = FrameBuffer(maxsize=2)
        self.stop_event = Event()

        # 启动推理线程
        self.inference_worker = InferenceWorker(self.buffer, self.stop_event)
        self.inference_worker.start()

        # 启动摄像头线程
        self.camera_worker = CameraWorker(
            camera_id, resolution, self.buffer, self.stop_event
        )
        self.camera_worker.start()

        QTimer.singleShot(100, self._check_workers_ready)

    def _check_workers_ready(self):
        """检查工作线程是否就绪"""
        camera_ready = self.camera_worker and self.camera_worker.ready.is_set()
        inference_ready = self.inference_worker and self.inference_worker.ready.is_set()

        if not camera_ready or not inference_ready:
            QTimer.singleShot(100, self._check_workers_ready)
            return

        if self.camera_worker.error:
            self.video_label.setText(f"摄像头错误: {self.camera_worker.error}")
            return

        if self.inference_worker.error:
            self.video_label.setText(f"模型加载错误: {self.inference_worker.error}")
            return

        self.display_timer.start(33)
        self.status_signal.emit("识别中...")

    def stop_camera(self):
        """停止摄像头"""
        self.display_timer.stop()

        if self.stop_event:
            self.stop_event.set()

        if self.camera_worker:
            self.camera_worker.join(timeout=1)
            self.camera_worker = None

        if self.inference_worker:
            self.inference_worker.join(timeout=1)
            self.inference_worker = None

        self.buffer = None
        self.stop_event = None
        self.video_label.setText("摄像头已停止")

    def _update_display(self):
        """更新显示（由定时器触发，在UI线程）"""
        if not self.buffer:
            return

        frame, results = self.buffer.get_latest()
        if frame is None:
            return

        if results:
            self.last_results = results
            for r in results:
                # 只有真正的新签到且未发送过信号才触发
                person_id = r.get("person_id")
                if (
                    r.get("is_new_checkin")
                    and person_id
                    and person_id not in self._emitted_checkins
                ):
                    self._emitted_checkins.add(person_id)
                    self.checkin_signal.emit(r["name"], True)

        display_frame = self._draw_results(frame.copy(), self.last_results)

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

        if self.camera_worker:
            main_window = self.window()
            if hasattr(main_window, "lbl_fps"):
                main_window.lbl_fps.setText(f"FPS: {self.camera_worker.fps:.1f}")

    def _draw_results(self, frame: np.ndarray, results: list) -> np.ndarray:
        """绘制识别结果"""
        for r in results:
            bbox = r.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            name = r.get("name", "Unknown")
            similarity = r.get("similarity", 0)
            is_new = r.get("is_new_checkin", False)
            is_real = r.get("is_real", True)
            spoof_type = r.get("spoof_type", None)

            # 根据状态选择颜色
            if not is_real:
                # 假脸检测 - 红色警告
                color = (0, 0, 255)
                label = f"⚠️ {spoof_type}"
            elif name == "Unknown":
                color = (128, 128, 128)  # 灰色表示未知
                label = "未知人员"
            elif is_new:
                color = (0, 255, 255)  # 黄色表示新签到
                label = f"{name} ({similarity:.2f}) ✓签到"
            else:
                color = (0, 255, 0)  # 绿色表示已识别
                label = f"{name} ({similarity:.2f})"

            # 绘制边框
            thickness = 3 if not is_real else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            self._put_chinese_text(frame, label, (x1, y1 - 25), color)

        return frame

        return frame

    def _put_chinese_text(
        self, img: np.ndarray, text: str, position: tuple, bg_color: tuple
    ):
        """绘制中文文本"""
        x, y = position
        h, w = img.shape[:2]

        text_w = len(text) * 12 + 10
        text_h = 25

        x = max(0, x)
        y = max(text_h, y)

        cv2.rectangle(img, (x, y - text_h), (x + text_w, y), bg_color, -1)

        if CHINESE_FONT:
            x1, y1 = max(0, x - 2), max(0, y - text_h - 2)
            x2, y2 = min(w, x + text_w + 2), min(h, y + 2)

            if x2 > x1 and y2 > y1:
                roi = img[y1:y2, x1:x2].copy()
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(roi_pil)
                draw.text(
                    (x - x1 + 2, y - text_h - y1 + 2),
                    text,
                    font=CHINESE_FONT,
                    fill=(255, 255, 255),
                )
                img[y1:y2, x1:x2] = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(
                img,
                text,
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def refresh_session(self):
        """刷新会话状态 - 在创建或结束会话后调用"""
        # 清空已发送签到信号集合
        self._emitted_checkins.clear()

        if self.inference_worker and self.inference_worker.attendance_service:
            # 清空当前会话引用，强制重新从数据库加载
            self.inference_worker.attendance_service.current_session = None
            # 清空已识别人员缓存
            self.inference_worker.attendance_service._recognized_persons.clear()
            # 重新获取活跃会话（会从数据库重新查询）
            self.inference_worker.attendance_service.get_active_session()

    def set_anti_spoof_enabled(self, enabled: bool):
        """设置是否启用活体检测"""
        if self.inference_worker and self.inference_worker.attendance_service:
            self.inference_worker.attendance_service.set_anti_spoof_enabled(enabled)

    def take_screenshot(self):
        """截图"""
        if self.buffer and self.buffer._latest_frame is not None:
            timestamp = int(time.time())
            path = Path(f"screenshot_{timestamp}.jpg")
            frame = self._draw_results(
                self.buffer._latest_frame.copy(), self.last_results
            )
            cv2.imwrite(str(path), frame)
            print(f"截图保存到: {path}")
            return str(path)
        return None

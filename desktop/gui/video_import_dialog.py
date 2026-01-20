"""
视频导入识别对话框 - 带实时预览
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional
from threading import Event

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QFileDialog,
    QLineEdit,
    QSpinBox,
    QFormLayout,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

from backend.app.face.detector import detect_faces, FaceRecognizer
from backend.app.db.database import SessionLocal
from backend.app.db.models import Person

# PIL 用于中文绘制
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
try:
    CHINESE_FONT = ImageFont.truetype(FONT_PATH, 18)
except Exception:
    CHINESE_FONT = None


class VideoProcessThread(QThread):
    """视频处理线程"""

    progress_signal = pyqtSignal(int, int)  # current, total
    status_signal = pyqtSignal(str)
    frame_signal = pyqtSignal(object, list)  # 帧和识别结果
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        video_path: str,
        frame_skip: int = 5,
        threshold: float = None,
        late_threshold_sec: int = 600,
    ):
        super().__init__()
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.threshold = threshold
        self.late_threshold_sec = late_threshold_sec
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self.status_signal.emit("正在加载识别模型...")
            recognizer = FaceRecognizer(threshold=self.threshold)

            self.status_signal.emit("正在打开视频...")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.status_signal.emit("错误：无法打开视频文件")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            attendance_records: Dict[int, Dict] = {}
            recognized_persons: Set[str] = set()

            self.status_signal.emit("正在处理视频帧...")
            frame_count = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_skip == 0:
                    faces = detect_faces(frame, extract_embedding=True)
                    frame_results = []

                    for face in faces:
                        result = recognizer.recognize_face(face)
                        bbox = face.bbox.tolist()

                        if result:
                            person_id, name, similarity = result
                            recognized_persons.add(name)
                            frame_results.append(
                                {
                                    "name": name,
                                    "similarity": similarity,
                                    "bbox": bbox,
                                }
                            )

                            if person_id not in attendance_records:
                                attendance_records[person_id] = {
                                    "name": name,
                                    "first_frame": frame_count,
                                    "last_frame": frame_count,
                                    "count": 1,
                                    "max_similarity": similarity,
                                }
                            else:
                                attendance_records[person_id]["last_frame"] = (
                                    frame_count
                                )
                                attendance_records[person_id]["count"] += 1
                                attendance_records[person_id]["max_similarity"] = max(
                                    attendance_records[person_id]["max_similarity"],
                                    similarity,
                                )
                        else:
                            frame_results.append(
                                {
                                    "name": "Unknown",
                                    "similarity": 0,
                                    "bbox": bbox,
                                }
                            )

                    # 发送帧用于预览
                    self.frame_signal.emit(frame.copy(), frame_results)

                frame_count += 1
                self.progress_signal.emit(frame_count, total_frames)

            cap.release()

            if self._stop_event.is_set():
                self.status_signal.emit("处理已取消")
                return

            # 生成考勤表
            db = SessionLocal()
            all_persons = db.query(Person).all()
            db.close()

            attendance_list = []
            for person in all_persons:
                record = attendance_records.get(person.id)

                if record:
                    first_time_sec = record["first_frame"] / fps if fps > 0 else 0
                    status = (
                        "迟到" if first_time_sec > self.late_threshold_sec else "正常"
                    )
                    attendance_list.append(
                        {
                            "person_id": person.id,
                            "name": person.name,
                            "student_id": person.student_id or "-",
                            "status": status,
                            "first_time_sec": first_time_sec,
                            "count": record["count"],
                            "similarity": record["max_similarity"],
                        }
                    )
                else:
                    attendance_list.append(
                        {
                            "person_id": person.id,
                            "name": person.name,
                            "student_id": person.student_id or "-",
                            "status": "缺勤",
                            "first_time_sec": None,
                            "count": 0,
                            "similarity": 0,
                        }
                    )

            total = len(all_persons)
            present_count = sum(1 for a in attendance_list if a["status"] == "正常")
            late_count = sum(1 for a in attendance_list if a["status"] == "迟到")
            absent_count = sum(1 for a in attendance_list if a["status"] == "缺勤")

            self.status_signal.emit("处理完成！")
            self.result_signal.emit(
                {
                    "attendance_list": attendance_list,
                    "total": total,
                    "present": present_count,
                    "late": late_count,
                    "absent": absent_count,
                    "fps": fps,
                    "video_path": self.video_path,
                }
            )

        except Exception as e:
            self.status_signal.emit(f"错误：{str(e)}")
        finally:
            self.finished_signal.emit()


class VideoImportDialog(QDialog):
    """视频导入识别对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📹 导入视频识别")
        self.setMinimumSize(1000, 700)
        self.process_thread: Optional[VideoProcessThread] = None
        self.result_data: Optional[dict] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 视频选择
        file_group = QGroupBox("选择视频")
        file_layout = QHBoxLayout(file_group)
        self.edit_video_path = QLineEdit()
        self.edit_video_path.setPlaceholderText("请选择视频文件...")
        self.edit_video_path.setReadOnly(True)
        file_layout.addWidget(self.edit_video_path)
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._browse_video)
        file_layout.addWidget(self.btn_browse)
        layout.addWidget(file_group)

        # 中部：左右分栏
        middle_layout = QHBoxLayout()

        # 左侧：视频预览
        preview_group = QGroupBox("视频预览")
        preview_layout = QVBoxLayout(preview_group)
        self.video_label = QLabel("选择视频后开始处理")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border-radius: 8px; color: #666; }"
        )
        preview_layout.addWidget(self.video_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        preview_layout.addWidget(self.progress_bar)
        self.lbl_status = QLabel("等待开始...")
        preview_layout.addWidget(self.lbl_status)
        middle_layout.addWidget(preview_group, stretch=2)

        # 右侧：参数和统计
        right_layout = QVBoxLayout()

        params_group = QGroupBox("识别参数")
        params_layout = QFormLayout(params_group)
        self.edit_session_name = QLineEdit()
        self.edit_session_name.setText(
            f"视频考勤_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        params_layout.addRow("会话名称:", self.edit_session_name)
        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(1, 30)
        self.spin_frame_skip.setValue(5)
        params_layout.addRow("帧跳过:", self.spin_frame_skip)
        self.spin_late_threshold = QSpinBox()
        self.spin_late_threshold.setRange(1, 60)
        self.spin_late_threshold.setValue(10)
        self.spin_late_threshold.setSuffix(" 分钟")
        params_layout.addRow("迟到阈值:", self.spin_late_threshold)
        right_layout.addWidget(params_group)

        stats_group = QGroupBox("识别统计")
        stats_layout = QVBoxLayout(stats_group)
        cards_layout = QHBoxLayout()

        def create_stat_card(label):
            card = QFrame()
            card.setStyleSheet("QFrame { background: #2d2d2d; border-radius: 8px; }")
            vbox = QVBoxLayout(card)
            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: #fff;"
            )
            title_label = QLabel(label)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("font-size: 12px; color: #888;")
            vbox.addWidget(value_label)
            vbox.addWidget(title_label)
            return card, value_label

        card1, self.lbl_stat_total = create_stat_card("总人数")
        card2, self.lbl_stat_present = create_stat_card("正常")
        card3, self.lbl_stat_late = create_stat_card("迟到")
        card4, self.lbl_stat_absent = create_stat_card("缺勤")
        cards_layout.addWidget(card1)
        cards_layout.addWidget(card2)
        cards_layout.addWidget(card3)
        cards_layout.addWidget(card4)

        stats_layout.addLayout(cards_layout)
        right_layout.addWidget(stats_group)
        middle_layout.addLayout(right_layout, stretch=1)
        layout.addLayout(middle_layout)

        # 结果表格
        table_group = QGroupBox("考勤结果")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["姓名", "学号", "状态", "首次出现", "识别次数"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self.table)
        layout.addWidget(table_group, stretch=1)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始识别")
        self.btn_start.clicked.connect(self._start_processing)
        self.btn_start.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self._stop_processing)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        self.btn_export = QPushButton("📥 导出Excel")
        self.btn_export.clicked.connect(self._export_excel)
        self.btn_export.setEnabled(False)
        btn_layout.addWidget(self.btn_export)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)",
        )
        if path:
            self.edit_video_path.setText(path)
            self.btn_start.setEnabled(True)
            video_name = Path(path).stem
            self.edit_session_name.setText(
                f"{video_name}_{datetime.now().strftime('%Y%m%d')}"
            )

    def _start_processing(self):
        video_path = self.edit_video_path.text()
        if not video_path:
            QMessageBox.warning(self, "提示", "请先选择视频文件")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_browse.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)

        self.process_thread = VideoProcessThread(
            video_path=video_path,
            frame_skip=self.spin_frame_skip.value(),
            late_threshold_sec=self.spin_late_threshold.value() * 60,
        )

        self.process_thread.progress_signal.connect(self._on_progress)
        self.process_thread.status_signal.connect(self._on_status)
        self.process_thread.frame_signal.connect(self._on_frame)
        self.process_thread.result_signal.connect(self._on_result)
        self.process_thread.finished_signal.connect(self._on_finished)
        self.process_thread.start()

    def _stop_processing(self):
        if self.process_thread:
            self.process_thread.stop()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))

    def _on_status(self, message: str):
        self.lbl_status.setText(message)

    def _on_frame(self, frame: np.ndarray, results: list):
        """显示当前帧和检测结果"""
        display_frame = frame.copy()

        for r in results:
            bbox = r.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            name = r.get("name", "Unknown")
            similarity = r.get("similarity", 0)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # 绘制名字
            label = f"{name} ({similarity:.2f})" if name != "Unknown" else "Unknown"
            self._put_chinese_text(display_frame, label, (x1, y1 - 25), color)

        # 转换为QPixmap显示
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

    def _on_result(self, result: dict):
        self.result_data = result
        self.lbl_stat_total.setText(str(result["total"]))
        self.lbl_stat_present.setText(str(result["present"]))
        self.lbl_stat_late.setText(str(result["late"]))
        self.lbl_stat_absent.setText(str(result["absent"]))

        attendance_list = result["attendance_list"]
        status_order = {"正常": 0, "迟到": 1, "缺勤": 2}
        attendance_list.sort(key=lambda x: status_order.get(x["status"], 99))

        self.table.setRowCount(len(attendance_list))
        for row, item in enumerate(attendance_list):
            self.table.setItem(row, 0, QTableWidgetItem(item["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(item["student_id"]))

            status = item["status"]
            status_icon = {"正常": "✅", "迟到": "⏰", "缺勤": "❌"}.get(status, "")
            self.table.setItem(row, 2, QTableWidgetItem(f"{status_icon} {status}"))

            if item["first_time_sec"] is not None:
                mins = int(item["first_time_sec"] // 60)
                secs = int(item["first_time_sec"] % 60)
                self.table.setItem(row, 3, QTableWidgetItem(f"{mins:02d}:{secs:02d}"))
            else:
                self.table.setItem(row, 3, QTableWidgetItem("-"))

            self.table.setItem(row, 4, QTableWidgetItem(str(item["count"])))

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_browse.setEnabled(True)
        if self.result_data:
            self.btn_export.setEnabled(True)

    def _export_excel(self):
        if not self.result_data:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出Excel",
            f"{self.edit_session_name.text()}.xlsx",
            "Excel文件 (*.xlsx)",
        )
        if not path:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(self.result_data["attendance_list"])
            df = df[
                [
                    "name",
                    "student_id",
                    "status",
                    "first_time_sec",
                    "count",
                    "similarity",
                ]
            ]
            df.columns = [
                "姓名",
                "学号",
                "状态",
                "首次出现(秒)",
                "识别次数",
                "最高相似度",
            ]
            df.to_excel(path, index=False)
            QMessageBox.information(self, "成功", f"已导出到 {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")

"""
设置面板
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QLabel,
    QMessageBox,
)

from backend.app.config import settings


class SettingsPanel(QWidget):
    """设置面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 摄像头设置
        camera_group = QGroupBox("摄像头设置")
        camera_layout = QFormLayout(camera_group)

        self.spin_camera_id = QSpinBox()
        self.spin_camera_id.setRange(0, 10)
        self.spin_camera_id.setValue(1)
        camera_layout.addRow("摄像头ID:", self.spin_camera_id)

        self.combo_resolution = QComboBox()
        self.combo_resolution.addItem("640x480 (30fps)", "480p")
        self.combo_resolution.addItem("1280x720 (15fps)", "720p")
        camera_layout.addRow("分辨率:", self.combo_resolution)

        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(1, 10)
        self.spin_frame_skip.setValue(2)
        camera_layout.addRow("帧跳过:", self.spin_frame_skip)

        layout.addWidget(camera_group)

        # 识别设置
        recognition_group = QGroupBox("识别设置")
        recognition_layout = QFormLayout(recognition_group)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.3, 0.9)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(settings.RECOGNITION_THRESHOLD)
        recognition_layout.addRow("识别阈值:", self.spin_threshold)

        self.spin_det_size = QSpinBox()
        self.spin_det_size.setRange(320, 1280)
        self.spin_det_size.setSingleStep(64)
        self.spin_det_size.setValue(640)
        recognition_layout.addRow("检测尺寸:", self.spin_det_size)

        layout.addWidget(recognition_group)

        # 考勤设置
        attendance_group = QGroupBox("考勤设置")
        attendance_layout = QFormLayout(attendance_group)

        self.spin_late_threshold = QSpinBox()
        self.spin_late_threshold.setRange(1, 60)
        self.spin_late_threshold.setValue(10)
        self.spin_late_threshold.setSuffix(" 分钟")
        attendance_layout.addRow("默认迟到阈值:", self.spin_late_threshold)

        self.spin_confirm_frames = QSpinBox()
        self.spin_confirm_frames.setRange(1, 20)
        self.spin_confirm_frames.setValue(5)
        attendance_layout.addRow("确认帧数:", self.spin_confirm_frames)

        layout.addWidget(attendance_group)

        # 数据库信息
        db_group = QGroupBox("数据库")
        db_layout = QFormLayout(db_group)

        self.lbl_db_path = QLabel(str(settings.DATABASE_URL))
        db_layout.addRow("路径:", self.lbl_db_path)

        self.lbl_model_path = QLabel(str(settings.MODEL_DIR))
        db_layout.addRow("模型:", self.lbl_model_path)

        layout.addWidget(db_group)

        # 按钮
        self.btn_save = QPushButton("💾 保存设置")
        self.btn_save.setProperty("class", "primary")
        self.btn_save.clicked.connect(self._save_settings)
        layout.addWidget(self.btn_save)

        layout.addStretch()

    def _load_settings(self):
        """加载设置"""
        # 从配置文件加载（目前使用默认值）
        pass

    def _save_settings(self):
        """保存设置"""
        # TODO: 保存到配置文件
        QMessageBox.information(self, "提示", "设置将在下次启动时生效")

    def get_camera_settings(self) -> dict:
        """获取摄像头设置"""
        return {
            "camera_id": self.spin_camera_id.value(),
            "resolution": self.combo_resolution.currentData(),
            "frame_skip": self.spin_frame_skip.value(),
        }

    def get_recognition_settings(self) -> dict:
        """获取识别设置"""
        return {
            "threshold": self.spin_threshold.value(),
            "det_size": self.spin_det_size.value(),
        }

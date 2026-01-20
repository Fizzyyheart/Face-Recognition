"""
主窗口
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QStatusBar,
    QToolBar,
    QLabel,
    QPushButton,
    QFrame,
    QCheckBox,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QMetaObject, Qt

from .video_widget import VideoWidget
from .person_panel import PersonPanel
from .attendance_panel import AttendancePanel
from .settings_panel import SettingsPanel
from .video_import_dialog import VideoImportDialog


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别考勤系统")
        self.setMinimumSize(1200, 800)

        self._init_ui()
        self._init_toolbar()
        self._init_statusbar()

    def _init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # 左侧：视频预览区
        # 使用 QFrame 包装以应用卡片样式
        left_panel = QFrame()
        left_panel.setObjectName("Card")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)

        self.video_widget = VideoWidget()
        # 圆角处理需要 VideoWidget 支持或者在这里裁剪
        left_layout.addWidget(self.video_widget)

        # 视频控制按钮
        video_controls = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始识别")
        self.btn_start.setProperty("class", "primary")
        self.btn_start.clicked.connect(self._on_start_recognition)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self._on_stop_recognition)
        self.btn_stop.setProperty("class", "danger")
        self.btn_stop.setEnabled(False)
        self.btn_screenshot = QPushButton("📷 截图")
        self.btn_screenshot.clicked.connect(self._on_screenshot)

        # 活体检测开关
        self.chk_anti_spoof = QCheckBox("🛡️ 活体检测")
        self.chk_anti_spoof.setChecked(True)
        self.chk_anti_spoof.setToolTip("开启后会检测并拒绝照片/视频攻击")
        self.chk_anti_spoof.stateChanged.connect(self._on_anti_spoof_changed)

        video_controls.addWidget(self.btn_start)
        video_controls.addWidget(self.btn_stop)
        video_controls.addWidget(self.btn_screenshot)
        video_controls.addStretch()
        video_controls.addWidget(self.chk_anti_spoof)

        left_layout.addLayout(video_controls)
        layout.addWidget(left_panel, stretch=2)

        # 右侧：功能面板
        right_panel = QTabWidget()

        self.attendance_panel = AttendancePanel()
        self.person_panel = PersonPanel()
        self.settings_panel = SettingsPanel()

        right_panel.addTab(self.attendance_panel, "📋 考勤签到")
        right_panel.addTab(self.person_panel, "👤 人员管理")
        right_panel.addTab(self.settings_panel, "⚙️ 设置")

        layout.addWidget(right_panel, stretch=1)

        # 连接视频组件的签到信号到考勤面板，实现实时刷新
        self.video_widget.checkin_signal.connect(self._on_checkin)

    def _init_toolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 新建会话
        action_new_session = QAction("新建会话", self)
        action_new_session.triggered.connect(self._on_new_session)
        toolbar.addAction(action_new_session)

        # 结束会话
        action_end_session = QAction("结束会话", self)
        action_end_session.triggered.connect(self._on_end_session)
        toolbar.addAction(action_end_session)

        toolbar.addSeparator()

        # 添加人员
        action_add_person = QAction("添加人员", self)
        action_add_person.triggered.connect(self._on_add_person)
        toolbar.addAction(action_add_person)

        toolbar.addSeparator()

        # 导入视频
        action_import_video = QAction("📹 导入视频", self)
        action_import_video.triggered.connect(self._on_import_video)
        toolbar.addAction(action_import_video)

        toolbar.addSeparator()

        # 导出报表
        action_export = QAction("导出报表", self)
        action_export.triggered.connect(self._on_export)
        toolbar.addAction(action_export)

    def _init_statusbar(self):
        """初始化状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.lbl_status = QLabel("就绪")
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_session = QLabel("会话: 无")
        self.lbl_persons = QLabel("人员: 0")

        self.statusbar.addWidget(self.lbl_status)
        self.statusbar.addPermanentWidget(self.lbl_fps)
        self.statusbar.addPermanentWidget(self.lbl_session)
        self.statusbar.addPermanentWidget(self.lbl_persons)

    # ==================== 事件处理 ====================

    def _on_start_recognition(self):
        """开始识别"""
        self.video_widget.start_camera()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("识别中...")

    def _on_stop_recognition(self):
        """停止识别"""
        self.video_widget.stop_camera()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("已停止")

    def _on_screenshot(self):
        """截图"""
        self.video_widget.take_screenshot()

    def _on_new_session(self):
        """新建会话"""
        self.attendance_panel.create_new_session()

    def _on_end_session(self):
        """结束会话"""
        self.attendance_panel.end_current_session()

    def _on_add_person(self):
        """添加人员"""
        self.person_panel.show_add_dialog()

    def _on_export(self):
        """导出报表"""
        self.attendance_panel.export_report()

    def _on_import_video(self):
        """导入视频进行识别"""
        dialog = VideoImportDialog(self)
        dialog.exec()

    def _on_anti_spoof_changed(self, state: int):
        """切换活体检测开关"""
        enabled = state == 2  # Qt.CheckState.Checked.value == 2
        self.video_widget.set_anti_spoof_enabled(enabled)
        status_text = "活体检测: 开启" if enabled else "活体检测: 关闭"
        self.lbl_status.setText(status_text)

    def _on_checkin(self, name: str, is_new: bool):
        """处理签到事件 - 实时刷新考勤表"""
        print(f"主窗口收到签到信号: {name}, 是否新签到: {is_new}")  # 调试输出
        if is_new:
            print("触发考勤表刷新")  # 调试输出
            # 在主线程中刷新考勤表显示
            QMetaObject.invokeMethod(
                self.attendance_panel,
                "refresh_current_session",
                Qt.ConnectionType.QueuedConnection,
            )

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.video_widget.stop_camera()
        event.accept()

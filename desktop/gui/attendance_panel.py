"""
考勤签到面板
"""

from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QDialog,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QMessageBox,
    QHeaderView,
    QGroupBox,
    QFileDialog,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot

from backend.app.db.database import SessionLocal
from backend.app.services.attendance_service import AttendanceService
from backend.app.db.models import AttendanceStatus


class NewSessionDialog(QDialog):
    """新建会话对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建考勤会话")
        self.setMinimumWidth(350)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("如: 机器视觉第3周")
        self.edit_name.setText(f"考勤_{datetime.now().strftime('%Y%m%d_%H%M')}")
        form.addRow("会话名称:", self.edit_name)

        self.spin_late = QSpinBox()
        self.spin_late.setRange(1, 60)
        self.spin_late.setValue(10)
        self.spin_late.setSuffix(" 分钟")
        form.addRow("迟到阈值:", self.spin_late)

        layout.addLayout(form)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("开始")
        self.btn_ok.setProperty("class", "primary")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def get_data(self) -> dict:
        return {
            "name": self.edit_name.text().strip(),
            "late_after_sec": self.spin_late.value() * 60,
        }


class AttendancePanel(QWidget):
    """考勤签到面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._load_sessions()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 当前会话信息
        session_group = QGroupBox("当前会话")
        session_layout = QVBoxLayout(session_group)

        self.lbl_session_name = QLabel("无活跃会话")
        self.lbl_session_name.setProperty("class", "h2")
        session_layout.addWidget(self.lbl_session_name)

        # 统计卡片区域
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(0, 10, 0, 10)
        stats_layout.setSpacing(10)

        # 辅助函数：创建统计卡片
        def create_stat_card(label, value_label, color_class=""):
            card = QFrame()
            card.setObjectName("StatCard")
            vbox = QVBoxLayout(card)
            vbox.setContentsMargins(5, 10, 5, 10)
            vbox.setSpacing(2)

            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setProperty("class", f"stat-value {color_class}")

            lbl_title = QLabel(label)
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_title.setProperty("class", "stat-label")

            vbox.addWidget(value_label)
            vbox.addWidget(lbl_title)
            return card

        # 初始化统计Label
        self.lbl_checked_in = QLabel("0")
        self.lbl_present = QLabel("0")
        self.lbl_late = QLabel("0")
        self.lbl_absent = QLabel("0")

        # 添加到布局
        stats_layout.addWidget(create_stat_card("已签到", self.lbl_checked_in))
        stats_layout.addWidget(create_stat_card("正常", self.lbl_present, "success"))
        stats_layout.addWidget(create_stat_card("迟到", self.lbl_late, "warning"))
        stats_layout.addWidget(create_stat_card("缺勤", self.lbl_absent, "danger"))

        session_layout.addWidget(stats_container)

        # 会话控制按钮
        ctrl_layout = QHBoxLayout()
        self.btn_new_session = QPushButton("🆕 新建会话")
        self.btn_new_session.setProperty("class", "primary")
        self.btn_new_session.clicked.connect(self.create_new_session)
        self.btn_end_session = QPushButton("⏹ 结束会话")
        self.btn_end_session.setProperty("class", "danger")
        self.btn_end_session.clicked.connect(self.end_current_session)
        self.btn_end_session.setEnabled(False)
        ctrl_layout.addWidget(self.btn_new_session)
        ctrl_layout.addWidget(self.btn_end_session)
        ctrl_layout.addStretch()

        session_layout.addLayout(ctrl_layout)
        layout.addWidget(session_group)

        # 历史会话选择
        history_layout = QHBoxLayout()
        history_layout.addWidget(QLabel("查看会话:"))
        self.combo_sessions = QComboBox()
        self.combo_sessions.currentIndexChanged.connect(self._on_session_selected)
        history_layout.addWidget(self.combo_sessions, stretch=1)
        self.btn_delete_session = QPushButton("🗑️ 删除")
        self.btn_delete_session.setProperty("class", "danger")
        self.btn_delete_session.clicked.connect(self.delete_session)
        history_layout.addWidget(self.btn_delete_session)
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.clicked.connect(self._load_sessions)
        history_layout.addWidget(self.btn_refresh)
        layout.addLayout(history_layout)

        # 签到列表
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["姓名", "学号", "状态", "签到时间", "识别"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        layout.addWidget(self.table)

    def _load_sessions(self):
        """加载会话列表"""
        self.combo_sessions.clear()

        try:
            db = SessionLocal()
            service = AttendanceService(db)
            sessions = service.get_all_sessions()

            active_session_index = -1
            for i, session in enumerate(sessions):
                status = "🟢" if session.is_active else "⚪"
                text = f"{status} {session.name}"
                self.combo_sessions.addItem(text, session.id)

                if session.is_active:
                    self.lbl_session_name.setText(f"📋 {session.name}")
                    self.btn_end_session.setEnabled(True)
                    self.btn_new_session.setEnabled(False)
                    active_session_index = i  # 记录活跃会话的索引

            # 如果有活跃会话，自动选中并加载其考勤数据
            if active_session_index >= 0:
                self.combo_sessions.setCurrentIndex(active_session_index)
                # 会自动触发_on_session_selected，加载考勤数据

            db.close()
        except Exception as e:
            print(f"加载会话失败: {e}")

    def _on_session_selected(self, index: int):
        """选择会话"""
        if index < 0:
            return

        session_id = self.combo_sessions.itemData(index)
        self._load_attendance(session_id)

    @pyqtSlot()
    def refresh_current_session(self):
        """刷新当前选中的会话考勤数据 - 用于实时更新"""
        print("考勤面板刷新被调用")  # 调试输出
        current_index = self.combo_sessions.currentIndex()
        print(
            f"当前下拉框索引: {current_index}, 总数: {self.combo_sessions.count()}"
        )  # 调试输出

        if current_index >= 0:
            session_id = self.combo_sessions.itemData(current_index)
            print(f"刷新会话 ID: {session_id}")  # 调试输出
            if session_id:
                self._load_attendance(session_id)
        else:
            print("没有选中的会话，尝试查找活跃会话")
            # 如果没有选中任何会话，尝试查找并加载活跃会话
            try:
                db = SessionLocal()
                service = AttendanceService(db)
                active_session = service.get_active_session()
                if active_session:
                    print(f"找到活跃会话: {active_session.id}")
                    self._load_attendance(active_session.id)
                db.close()
            except Exception as e:
                print(f"查找活跃会话失败: {e}")

    def _load_attendance(self, session_id: int):
        """加载考勤记录"""
        self.table.setRowCount(0)

        try:
            db = SessionLocal()
            service = AttendanceService(db)
            data = service.get_session_attendance(session_id)

            if not data:
                db.close()
                return

            # 更新统计
            self.lbl_checked_in.setText(str(data["checked_in"]))
            self.lbl_present.setText(str(data["present"]))
            self.lbl_late.setText(str(data["late"]))
            self.lbl_absent.setText(str(data["absent"]))

            # 填充表格
            for att in data["attendance_list"]:
                row = self.table.rowCount()
                self.table.insertRow(row)

                self.table.setItem(row, 0, QTableWidgetItem(att["name"]))
                self.table.setItem(row, 1, QTableWidgetItem(att["student_id"] or "-"))

                # 状态
                status = att["status"]
                if status == AttendanceStatus.PRESENT.value:
                    status_text = "✅ 正常"
                elif status == AttendanceStatus.LATE.value:
                    status_text = "⚠️ 迟到"
                else:
                    status_text = "❌ 缺勤"
                self.table.setItem(row, 2, QTableWidgetItem(status_text))

                # 时间
                time_str = att["first_seen"][:19] if att["first_seen"] else "-"
                self.table.setItem(row, 3, QTableWidgetItem(time_str))

                # 识别确认（每人只打卡一次）
                self.table.setItem(row, 4, QTableWidgetItem("✓ 已确认"))

            db.close()
        except Exception as e:
            print(f"加载考勤失败: {e}")

    def create_new_session(self):
        """创建新会话"""
        dialog = NewSessionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()

            if not data["name"]:
                QMessageBox.warning(self, "错误", "会话名称不能为空")
                return

            try:
                db = SessionLocal()
                service = AttendanceService(db)
                session = service.create_session(
                    name=data["name"],
                    late_after_sec=data["late_after_sec"],
                )
                db.close()

                self.lbl_session_name.setText(f"📋 {session.name}")
                self.btn_end_session.setEnabled(True)
                self.btn_new_session.setEnabled(False)

                # 通知video_widget刷新会话
                main_window = self.window()
                if hasattr(main_window, "video_widget"):
                    main_window.video_widget.refresh_session()

                QMessageBox.information(self, "成功", f"会话 '{session.name}' 已开始")
                self._load_sessions()
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))

    def end_current_session(self):
        """结束当前会话"""
        reply = QMessageBox.question(
            self,
            "确认",
            "确定要结束当前会话吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                db = SessionLocal()
                service = AttendanceService(db)
                success, msg = service.end_session()
                db.close()

                if success:
                    self.lbl_session_name.setText("无活跃会话")
                    self.btn_end_session.setEnabled(False)
                    self.btn_new_session.setEnabled(True)

                    # 通知video_widget刷新会话
                    main_window = self.window()
                    if hasattr(main_window, "video_widget"):
                        main_window.video_widget.refresh_session()

                    QMessageBox.information(self, "成功", msg)
                    self._load_sessions()
                else:
                    QMessageBox.warning(self, "失败", msg)
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))

    def delete_session(self):
        """删除会话"""
        session_id = self.combo_sessions.currentData()
        if not session_id:
            QMessageBox.warning(self, "错误", "请先选择一个会话")
            return

        # 获取会话名称
        session_text = self.combo_sessions.currentText()

        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除会话 '{session_text}' 及其所有考勤记录吗？\n\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                db = SessionLocal()
                service = AttendanceService(db)
                success, msg = service.delete_session(session_id)
                db.close()

                if success:
                    QMessageBox.information(self, "成功", msg)
                    self._load_sessions()
                else:
                    QMessageBox.warning(self, "失败", msg)
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))

    def export_report(self):
        """导出考勤报表"""
        session_id = self.combo_sessions.currentData()
        if not session_id:
            QMessageBox.warning(self, "错误", "请先选择一个会话")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出报表", f"attendance_{session_id}.csv", "CSV文件 (*.csv)"
        )

        if path:
            try:
                db = SessionLocal()
                service = AttendanceService(db)
                data = service.get_session_attendance(session_id)
                db.close()

                # 写入 CSV
                with open(path, "w", encoding="utf-8-sig") as f:
                    f.write("姓名,学号,状态,签到时间\n")
                    for att in data["attendance_list"]:
                        f.write(
                            f"{att['name']},{att['student_id'] or ''},{att['status']},"
                            f"{att['first_seen'] or ''}\n"
                        )

                QMessageBox.information(self, "成功", f"报表已导出到: {path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))

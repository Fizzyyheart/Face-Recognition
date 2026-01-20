"""
人员管理面板
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLineEdit,
    QLabel,
    QDialog,
    QFormLayout,
    QFileDialog,
    QMessageBox,
    QHeaderView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from backend.app.db.database import SessionLocal
from backend.app.services.person_service import PersonService


class AddPersonDialog(QDialog):
    """添加人员对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加人员")
        self.setMinimumWidth(400)
        self.image_path: Optional[str] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("必填")
        form.addRow("姓名:", self.edit_name)

        self.edit_student_id = QLineEdit()
        self.edit_student_id.setPlaceholderText("可选")
        form.addRow("学号:", self.edit_student_id)

        # 图片选择
        image_layout = QHBoxLayout()
        self.lbl_image = QLabel("未选择")
        self.btn_select_image = QPushButton("选择照片")
        self.btn_select_image.clicked.connect(self._select_image)
        image_layout.addWidget(self.lbl_image)
        image_layout.addWidget(self.btn_select_image)
        form.addRow("人脸照片:", image_layout)

        # 图片预览
        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(150, 150)
        self.lbl_preview.setStyleSheet("border: 1px solid #ccc;")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form.addRow("", self.lbl_preview)

        layout.addLayout(form)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.setProperty("class", "primary")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择人脸照片", "", "图片文件 (*.jpg *.jpeg *.png)"
        )
        if path:
            self.image_path = path
            self.lbl_image.setText(path.split("/")[-1])

            # 显示预览
            pixmap = QPixmap(path)
            scaled = pixmap.scaled(
                150,
                150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.lbl_preview.setPixmap(scaled)

    def get_data(self) -> dict:
        return {
            "name": self.edit_name.text().strip(),
            "student_id": self.edit_student_id.text().strip() or None,
            "image_path": self.image_path,
        }


class PersonPanel(QWidget):
    """人员管理面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._load_persons()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 工具栏
        toolbar = QHBoxLayout()

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索人员...")
        self.search_edit.textChanged.connect(self._on_search)
        toolbar.addWidget(self.search_edit)

        self.btn_add = QPushButton("➕ 添加")
        self.btn_add.setProperty("class", "primary")
        self.btn_add.clicked.connect(self.show_add_dialog)
        toolbar.addWidget(self.btn_add)

        self.btn_refresh = QPushButton("🔄 刷新")
        self.btn_refresh.clicked.connect(self._load_persons)
        toolbar.addWidget(self.btn_refresh)

        layout.addLayout(toolbar)

        # 人员表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["姓名", "学号", "样本数", "操作"])

        # 设置表格样式，增加padding
        self.table.setStyleSheet("""
            QTableWidget::item {
                padding: 5px 10px;
            }
        """)

        # 设置列宽：学号、样本数固定宽度，姓名自适应，操作固定
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # 姓名
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)  # 学号
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # 样本数
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)  # 操作

        self.table.setColumnWidth(1, 100)  # 学号列宽
        self.table.setColumnWidth(2, 120)  # 样本数列宽（增加到120）
        self.table.setColumnWidth(3, 100)  # 操作列宽

        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        layout.addWidget(self.table)

    def _load_persons(self):
        """加载人员列表"""
        self.table.setRowCount(0)

        try:
            db = SessionLocal()
            service = PersonService(db)
            persons = service.get_all_persons()

            for person in persons:
                stats = service.get_person_stats(person.id)
                self._add_person_row(stats)

            db.close()
        except Exception as e:
            print(f"加载人员失败: {e}")

    def _add_person_row(self, stats: dict):
        """添加人员行"""
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(stats.get("name", "")))
        self.table.setItem(row, 1, QTableWidgetItem(stats.get("student_id", "") or "-"))

        # 样本数居中显示
        sample_item = QTableWidgetItem(str(stats.get("sample_count", 0)))
        sample_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 2, sample_item)

        # 操作按钮
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(2, 2, 2, 2)

        btn_delete = QPushButton("删除")
        btn_delete.setProperty("person_id", stats.get("id"))
        btn_delete.setProperty("class", "danger")
        btn_delete.clicked.connect(self._on_delete)
        btn_layout.addWidget(btn_delete)

        self.table.setCellWidget(row, 3, btn_widget)

    def _on_search(self, text: str):
        """搜索过滤"""
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            sid_item = self.table.item(row, 1)

            match = (
                text.lower() in name_item.text().lower()
                or text.lower() in sid_item.text().lower()
            )
            self.table.setRowHidden(row, not match)

    def show_add_dialog(self):
        """显示添加对话框"""
        dialog = AddPersonDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()

            if not data["name"]:
                QMessageBox.warning(self, "错误", "姓名不能为空")
                return

            try:
                db = SessionLocal()
                service = PersonService(db)
                success, msg, person = service.add_person(
                    name=data["name"],
                    student_id=data["student_id"],
                    face_image_path=data["image_path"],
                )
                db.close()

                if success:
                    QMessageBox.information(self, "成功", msg)
                    self._load_persons()
                else:
                    QMessageBox.warning(self, "失败", msg)
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))

    def _on_delete(self):
        """删除人员"""
        btn = self.sender()
        person_id = btn.property("person_id")

        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除该人员吗？此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                db = SessionLocal()
                service = PersonService(db)
                success, msg = service.delete_person(person_id)
                db.close()

                if success:
                    QMessageBox.information(self, "成功", msg)
                    self._load_persons()
                else:
                    QMessageBox.warning(self, "失败", msg)
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))

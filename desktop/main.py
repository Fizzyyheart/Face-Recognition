"""
人脸识别考勤系统 - PyQt6 桌面应用
"""

import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 添加 desktop 目录
DESKTOP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DESKTOP_ROOT))

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 加载样式表
    style_path = DESKTOP_ROOT / "assets" / "style.qss"
    if style_path.exists():
        with open(style_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    # 设置应用信息
    app.setApplicationName("人脸识别考勤系统")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("FaceRecognition")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

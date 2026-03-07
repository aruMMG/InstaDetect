import os

# Must be set before importing PyQt5 or any module that imports cv2/PyQt.
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

import sys
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Edge AI Defect Demo")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
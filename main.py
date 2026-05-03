import sys

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.styles import MODERN_STYLE


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(MODERN_STYLE)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

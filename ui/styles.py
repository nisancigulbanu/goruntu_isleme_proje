def build_modern_style(scale=1.0):
    def px(value):
        return max(1, int(round(value * scale)))

    return f"""
QMainWindow, QDialog {{ background-color: #14151c; }}
QWidget {{ color: #ececf3; font-family: 'Segoe UI', sans-serif; font-size: {px(14)}px; }}
QLabel {{ color: #dce0eb; }}
QLabel#panelMeta {{ padding: {px(12)}px {px(14)}px; background: #1e2030; border: 1px solid #2e3348; border-radius: {px(12)}px; }}
QLabel#panelImage {{ border: 1px dashed #3d4a63; border-radius: {px(14)}px; background: #1a1c28; padding: {px(10)}px; }}
QGroupBox#historyCard {{ border: 1px solid #2e3348; border-radius: {px(14)}px; margin-top: {px(10)}px; padding: {px(12)}px; background: #1e2030; }}
QWidget#tabPage {{ background: #181a26; }}
QGroupBox {{
    border: 1px solid #323848;
    border-radius: {px(12)}px;
    margin-top: {px(14)}px;
    padding: {px(16)}px {px(12)}px {px(12)}px {px(12)}px;
    background: #222534;
    font-size: {px(15)}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {px(12)}px;
    padding: 0 {px(4)}px;
}}
QPushButton {{
    background: #2a3145;
    border: 1px solid #3d4660;
    border-radius: {px(10)}px;
    padding: {px(10)}px {px(16)}px;
    min-height: {px(22)}px;
    color: #f4f6fb;
    font-size: {px(14)}px;
    font-weight: 600;
}}
QPushButton:hover {{ background: #343d56; border-color: #5d8cff; }}
QTabWidget::pane {{ border: 1px solid #2e3348; border-radius: {px(12)}px; background: #181a26; padding: {px(8)}px; }}
QTabBar::tab {{
    background: #1e2230;
    color: #8b92a8;
    padding: {px(12)}px {px(20)}px;
    min-width: {px(130)}px;
    border-top-left-radius: {px(10)}px;
    border-top-right-radius: {px(10)}px;
    font-size: {px(15)}px;
    font-weight: 600;
}}
QTabBar::tab:selected {{ background: #181a26; color: #fff; font-weight: 700; border-bottom: 3px solid #5d8cff; }}
QSlider::groove:horizontal {{ height: {px(8)}px; background: #12141d; border-radius: {px(4)}px; }}
QSlider::handle:horizontal {{
    background: #e8ebf4;
    border: 1px solid #5d8cff;
    width: {px(18)}px;
    height: {px(18)}px;
    margin: -{px(6)}px 0;
    border-radius: {px(9)}px;
}}
QListWidget {{
    background: #151721;
    border: 1px solid #2e3348;
    border-radius: {px(12)}px;
    font-size: {px(14)}px;
    padding: {px(4)}px;
}}
QListWidget::item {{ padding: {px(6)}px {px(8)}px; }}
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: #222534;
    border: 1px solid #3d4660;
    border-radius: {px(8)}px;
    padding: {px(8)}px {px(10)}px;
    min-height: {px(22)}px;
    color: #ececf3;
    font-size: {px(14)}px;
}}
QToolBar {{ background: #1a1c28; border-bottom: 1px solid #2e3348; padding: {px(10)}px; spacing: {px(8)}px; }}
QToolButton {{
    font-size: {px(14)}px;
    padding: {px(8)}px {px(12)}px;
    min-height: {px(24)}px;
}}
QFrame#controlCard {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #232738, stop:1 #1b1f2c);
    border: 1px solid #343b52;
    border-radius: {px(18)}px;
}}
QLabel#cardTitle {{
    font-size: {px(18)}px;
    font-weight: 700;
    color: #f5f7ff;
}}
QLabel#cardDescription {{
    font-size: {px(13)}px;
    color: #9ca6bf;
}}
QLabel#fieldLabel {{
    font-size: {px(13)}px;
    color: #aeb7cd;
    font-weight: 600;
}}
QLabel#valueBadge {{
    background: #141925;
    border: 1px solid #3a4766;
    border-radius: {px(12)}px;
    padding: {px(6)}px {px(10)}px;
    color: #f4f7ff;
    font-size: {px(13)}px;
    font-weight: 700;
    min-width: {px(52)}px;
}}
QPushButton#primaryButton {{
    background: #5d8cff;
    border: 1px solid #80a6ff;
    color: #ffffff;
}}
QPushButton#primaryButton:hover {{ background: #74a0ff; border-color: #9abbff; }}
QScrollArea#tabScroll {{ border: none; background: transparent; }}
QWidget#tabContent {{ background: transparent; }}
QScrollBar:vertical {{ background: #151721; width: {px(12)}px; }}
QScrollBar::handle:vertical {{ background: #3d455c; border-radius: {px(6)}px; min-height: {px(28)}px; }}
"""


MODERN_STYLE = build_modern_style()

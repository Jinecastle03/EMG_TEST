from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QLabel, QProgressBar

BG = "#121418"
PANEL = "#1b1e24"
TEXT = "#e6e6e6"
MUTED = "#8a919c"
ACCENT = "#4fa3ff"
OK = "#4caf50"
WARN = "#ffb300"
ERR = "#ef5350"
REPLAY = "#ff8f00"
LOW = "#4fc3f7"
HIGH = "#ff7043"
CH_COLORS = ["#ef5350", "#ffa726", "#ffee58", "#66bb6a", "#26c6da", "#42a5f5", "#ab47bc", "#ec407a"]
POSE_COLORS = ["#9e9e9e", "#66bb6a", "#42a5f5", "#ffca28", "#ab47bc", "#26c6da", "#ef5350", "#8d6e63", "#d4e157", "#ff7043"]

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QGroupBox {{ border: 1px solid #2c313a; border-radius: 6px; margin-top: 14px; padding: 6px; background: {PANEL}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {MUTED}; }}
QPushButton {{ background: #262b33; border: 1px solid #39404b; border-radius: 6px; padding: 8px 14px; font-weight: 600; }}
QPushButton:hover {{ background: #2f3540; }}
QPushButton:checked {{ background: #1f5fa8; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #59606b; background: #1d2026; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QListWidget, QTableWidget {{
    background: #0e1013; border: 1px solid #333943; border-radius: 4px; padding: 3px; }}
QHeaderView::section {{ background: #20242b; color: {MUTED}; border: none; padding: 4px; }}
QProgressBar {{ border: 1px solid #333943; border-radius: 4px; background: #0e1013; text-align: center; height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QTabBar::tab {{ background: #20242b; padding: 6px 12px; }}
QTabBar::tab:selected {{ background: #2c3340; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
"""


def apply(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, "#0e1013"),
                      (QPalette.AlternateBase, PANEL), (QPalette.Text, TEXT), (QPalette.Button, "#262b33"),
                      (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
                      (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT)):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def big_label(text="", px=48, color=TEXT, bold=True):
    """Large label. The size goes into the widget's own style sheet: the application style sheet sets
    font-size for QWidget, which would otherwise override QLabel.setFont()."""
    lab = QLabel(text)
    lab.setProperty("_px", px)
    lab.setProperty("_bold", bold)
    set_color(lab, color)
    return lab


def set_color(lab, color, bg=None):
    px, bold = lab.property("_px"), lab.property("_bold")
    font = (f" font-size: {px}px;" if px else "") + (" font-weight: 700;" if bold else "")
    lab.setStyleSheet(f"color: {color};{font}" + (f" background: {bg}; border-radius: 6px; padding: 4px;" if bg else ""))


def bar(color=ACCENT, maximum=100):
    b = QProgressBar()
    b.setRange(0, maximum)
    b.setTextVisible(True)
    b.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
    return b

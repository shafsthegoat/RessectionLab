"""Original restrained medical workspace styling."""

COLORS = [(230, 158, 83), (192, 119, 221), (78, 186, 215), (111, 194, 169)]
STYLE = """
QWidget { background: #101316; color: #e2e8e9; font-family: '.AppleSystemUIFont', 'Helvetica Neue', sans-serif; font-size: 12px; }
QMainWindow { background: #0a0c0e; }
QLabel { background: transparent; }
QLabel#brand { font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #f5f7f7; }
QLabel#section { color: #849195; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; padding-top: 12px; padding-bottom: 5px; }
QLabel#muted { color: #899699; font-size: 11px; }
QLabel#title { font-size: 17px; font-weight: 500; }
QLabel#badge { color: #a2c4ba; background: #182924; border: 1px solid #29483d; border-radius: 4px; padding: 4px 8px; font-size: 10px; }
QLabel#warning { color: #d4b486; background: #26231c; border: 1px solid #403829; border-radius: 5px; padding: 9px; }
QFrame#panel { background: #101619; border: 1px solid #273034; border-radius: 7px; }
QPushButton { background: #1c2529; border: 1px solid #334247; border-radius: 5px; padding: 8px 11px; color: #e1e8e9; }
QPushButton:hover { background: #293a3e; border-color: #6a9890; }
QPushButton:pressed { background: #364e4a; }
QPushButton:disabled { color: #566267; border-color: #273135; background: #161d20; }
QPushButton#primary { background: #87c5b5; color: #10251e; border: none; font-weight: 600; padding: 10px; }
QPushButton#primary:hover { background: #a4dece; }
QPushButton#primary:disabled { background: #314940; color: #6c8f82; }
QComboBox, QDoubleSpinBox { background: #171f23; border: 1px solid #334247; border-radius: 4px; padding: 6px; }
QComboBox QAbstractItemView { background: #172126; selection-background-color: #334e49; }
QCheckBox { spacing: 7px; padding: 5px 0; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #60716e; border-radius: 3px; background: #182226; }
QCheckBox::indicator:checked { background: #87c5b5; border-color: #87c5b5; }
QCheckBox:disabled { color: #657277; }
QListWidget { border: none; background: #11191c; outline: none; }
QListWidget::item { border: 1px solid #2b383d; border-radius: 5px; padding: 10px 8px; margin: 3px 0; }
QListWidget::item:selected { background: #243e37; border-color: #87c5b5; }
QTextBrowser, QPlainTextEdit { background: #11191c; border: 1px solid #28353a; border-radius: 5px; color: #bcc9cc; padding: 7px; }
QProgressBar { background: #1c292d; border: none; border-radius: 2px; height: 4px; text-align: center; }
QProgressBar::chunk { background: #87c5b5; }
QSlider::groove:horizontal { height: 3px; background: #344348; }
QSlider::handle:horizontal { background: #91cab9; width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; }
QSplitter::handle { background: #0b1012; width: 7px; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #11191c; width: 7px; }
QScrollBar::handle:vertical { background: #405351; border-radius: 3px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background: #11191c; color: #9dacad; border-top: 1px solid #293337; }
QMenuBar, QMenu { background: #101619; }
QMenu::item:selected { background: #2b443c; }
"""

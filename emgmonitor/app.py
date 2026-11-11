"""Entry point:  python -m emgmonitor.app"""
import sys


def main():
    from PySide6.QtWidgets import QApplication

    from .gui import theme
    from .gui.main_window import MainWindow
    try:
        import torch
        torch.set_num_threads(2)          # tiny CNNs; keep CPU free for the GUI / acquisition
    except Exception:
        pass
    app = QApplication(sys.argv)
    theme.apply(app)
    w = MainWindow()
    w.show()
    if "--replay-demo" in sys.argv:      # optional: start replaying the newest training session
        from pathlib import Path

        from .acquisition import ReplaySource
        from .session import list_sessions
        s = [x for x in list_sessions() if x["kind"] == "training"]
        if s:
            w.engine.start(ReplaySource(Path(s[-1]["path"])))
            w.btn["live"].setChecked(True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

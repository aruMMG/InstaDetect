from PyQt5.QtWidgets import QLabel


class StatusChip(QLabel):
    COLOR_MAP = {
        "idle": "#616161",
        "preparing dataset": "#455A64",
        "uploading dataset": "#5E35B1",
        "training started": "#1565C0",
        "training running": "#0277BD",
        "validation running": "#6A1B9A",
        "export running": "#EF6C00",
        "deployment running": "#00897B",
        "pulling model": "#00897B",
        "completed": "#2E7D32",
        "failed": "#C62828",
        "running": "#2E7D32",
        "stopped": "#757575",
        "warning": "#ED6C02",
    }

    def __init__(self, text: str = "idle", parent=None):
        super().__init__(parent)
        self.set_status(text)

    def set_status(self, text: str):
        key = text.strip().lower()
        color = self.COLOR_MAP.get(key, "#455A64")
        self.setText(text)
        self.setStyleSheet(
            f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 10px;
                padding: 4px 10px;
                font-weight: 600;
            }}
            """
        )

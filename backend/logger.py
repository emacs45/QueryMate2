import logging
import os

# Log-Verzeichnis erstellen, falls nicht vorhanden
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class Logger:
    def __init__(self, name="QueryMate", log_file="logs/app.log", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Log-Format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s: %(message)s")

        # Datei-Logging
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console-Logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

# Globale Logger-Instanz erstellen
app_logger = Logger().get_logger()
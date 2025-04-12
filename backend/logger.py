import logging
import os

def setup_logger(name="QueryMate", log_file="logs/app.log", level=logging.INFO):
    # Log-Verzeichnis erstellen, falls nicht vorhanden
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Verhindert doppelte Handler, falls Funktion mehrfach aufgerufen wird
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s: %(message)s")

        # Datei-Logging
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console-Logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Globale Logger-Instanz (wie zuvor)
app_logger = setup_logger()

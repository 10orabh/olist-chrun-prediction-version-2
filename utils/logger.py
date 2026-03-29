import logging
import os   
from datetime import datetime

class Logger:
    def __init__(self, module_name: str, level:str):
        self.module_name = module_name
        self.level = level
        self.logger = logging.getLogger(self.module_name)
        self.logger.setLevel(self.level)

        if not self.logger.handlers:
            self._setup_logger()

    def _setup_logger(self):
        log = 'logs'
        os.makedirs(log, exist_ok=True)
        LOG_DIR = os.path.join("logs", self.module_name)
        os.makedirs(LOG_DIR, exist_ok=True)


        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = f"{timestamp}.log"
        log_path = os.path.join(LOG_DIR, log_file)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger  
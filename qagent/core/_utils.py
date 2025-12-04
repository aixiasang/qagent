import base64
import logging
from typing import Optional


class AgentLogger:
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        enabled: bool = True,
        log_file: Optional[str] = None,
        format: Optional[str] = None,
    ):
        self.enabled = enabled
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()

        if not enabled:
            self.logger.addHandler(logging.NullHandler())
            return

        if format is None:
            format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

        formatter = logging.Formatter(format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.info(msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.debug(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.error(msg, **kwargs)

    def disable(self):
        self.enabled = False
        self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())

    def enable(self):
        self.enabled = True


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

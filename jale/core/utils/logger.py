import logging
from datetime import datetime
from pathlib import Path


def setup_logger(project_path: Path):
    """Initialize logging with a file handler in the project directory."""
    logger = logging.getLogger("ale_logger")

    # Prevent adding handlers multiple times
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_format)

        # File handler in the project directory
        start_time = datetime.now().strftime("%Y%m%d_%H%M")
        file_handler = logging.FileHandler(project_path / f"logs/{start_time}.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

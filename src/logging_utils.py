import os
import time
import logging


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "training.log",
    timezone: str = "Asia/Riyadh",
) -> logging.Logger:
    """
    Configure application-wide logging:
    - Creates logs directory if needed
    - Writes logs to logs/training.log
    - Uses local timezone (default: Asia/Riyadh)
    - Mirrors logs to console
    """
    # Configure timezone
    os.environ["TZ"] = timezone
    try:
        time.tzset()
    except AttributeError:
        # tzset may not exist on some platforms (e.g., Windows); ignore in that case
        pass

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove old handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Logging configured. Writing to %s", log_path)
    return logger
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from app.core.config import settings


def setup_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # 防止重复添加 Handler

    logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    log_dir = settings.log_dir or "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_path = os.path.join(log_dir, f"{name}.log")

    # 文件日志处理器（轮转）
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # 控制台日志处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if settings.debug else logging.WARNING)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 劫持 uvicorn/gunicorn 日志
    if name == "app":
        logging.getLogger("uvicorn.access").handlers = logger.handlers
        logging.getLogger("uvicorn.error").handlers = logger.handlers
        logging.getLogger("uvicorn").handlers = logger.handlers
        logging.getLogger("gunicorn.error").handlers = logger.handlers

    return logger


# 全局日志器实例
logger = setup_logger()

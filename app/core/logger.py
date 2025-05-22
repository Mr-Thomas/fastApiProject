import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from app.core.config import settings

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LevelFilter(logging.Filter):
    """只允许通过指定等级的日志"""

    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


def setup_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # 避免重复添加 handler

    logger.setLevel(logging.DEBUG)

    log_dir = settings.log_dir or "logs"
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 为每种日志级别添加一个 TimedRotatingFileHandler
    for level_name, level in LOG_LEVELS.items():
        file_path = os.path.join(log_dir, f"{name}_{level_name}.log")
        handler = TimedRotatingFileHandler(
            filename=file_path,
            when='midnight',
            interval=1,
            backupCount=7,  # 保留最近 7 天
            encoding='utf-8',
            utc=False  # 本地时间
        )
        handler.setLevel(level)
        handler.addFilter(LevelFilter(level))
        handler.setFormatter(formatter)
        # 设置后缀为日期格式
        handler.suffix = "%Y-%m-%d"
        logger.addHandler(handler)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if settings.debug else logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 劫持 uvicorn/gunicorn 日志
    if name == "app":
        for target in ["uvicorn.access", "uvicorn.error", "uvicorn", "gunicorn.error"]:
            logging.getLogger(target).handlers = logger.handlers

    return logger


# 初始化全局日志器
logger = setup_logger()

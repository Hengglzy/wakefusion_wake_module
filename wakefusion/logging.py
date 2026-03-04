"""
结构化日志模块
提供统一的日志接口，支持结构化输出和性能监控
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器（文本格式）

    统一输出格式：
        [Time] [Level] [Logger] -> Message
    """

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为简洁的文本格式"""
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        logger_name = record.name
        msg = record.getMessage()

        base = f"[{ts}] [{level}] [{logger_name}] -> {msg}"

        # 若存在异常信息，追加到日志末尾，便于人类阅读
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            base = f"{base}\n{exc_text}"

        return base


class WakeFusionLogger:
    """WakeFusion日志器"""

    def __init__(self, name: str = "wakefusion", level: str = "INFO"):
        """
        初始化日志器

        Args:
            name: 日志器名称
            level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.propagate = False

        # 清除现有handlers
        self.logger.handlers.clear()

        # 控制台handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def _log_with_context(self, level: int, msg: str, **kwargs):
        """带上下文的日志记录"""
        extra = kwargs.pop('extra', {})
        session_id = kwargs.pop('session_id', None)
        event_id = kwargs.pop('event_id', None)

        record = self.logger.makeRecord(
            self.logger.name, level, None, None, msg, (), None
        )

        if session_id:
            record.session_id = session_id
        if event_id:
            record.event_id = event_id
        if extra:
            record.extra = extra

        self.logger.handle(record)

    def debug(self, msg: str, **kwargs):
        """调试日志"""
        self._log_with_context(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """信息日志"""
        self._log_with_context(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """警告日志"""
        self._log_with_context(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """错误日志"""
        self._log_with_context(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """严重错误日志"""
        self._log_with_context(logging.CRITICAL, msg, **kwargs)

    @contextmanager
    def log_latency(self, operation: str, **context):
        """
        上下文管理器：自动记录操作耗时

        Usage:
            with logger.log_latency("kws_inference", model="openwakeword"):
                result = model.predict(frame)
        """
        import time
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.debug(
                f"{operation} completed",
                extra={
                    "operation": operation,
                    "latency_ms": round(elapsed_ms, 2),
                    **context
                }
            )


# 全局日志器实例
_global_logger: Optional[WakeFusionLogger] = None


def get_logger(name: str = "wakefusion", level: str = "INFO") -> WakeFusionLogger:
    """
    获取全局日志器实例

    Args:
        name: 日志器名称
        level: 日志级别

    Returns:
        WakeFusionLogger: 日志器实例
    """
    global _global_logger

    if _global_logger is None or _global_logger.logger.name != name:
        _global_logger = WakeFusionLogger(name, level)

    return _global_logger


def set_log_level(level: str):
    """动态设置日志级别"""
    logger = get_logger()
    logger.logger.setLevel(getattr(logging, level.upper()))

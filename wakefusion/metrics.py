"""
指标收集模块
提供性能指标监控和统计
"""

import time
import psutil
from threading import Lock
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricValue:
    """指标值"""
    value: float
    count: int = 1
    min: float = float('inf')
    max: float = float('-inf')
    sum: float = 0.0

    def update(self, value: float):
        """更新指标值"""
        self.value = value
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.sum += value

    @property
    def avg(self) -> float:
        """计算平均值"""
        return self.sum / self.count if self.count > 0 else 0.0


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        """初始化指标收集器"""
        self._metrics: Dict[str, MetricValue] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._lock = Lock()

    def record(self, name: str, value: float):
        """
        记录指标值（会统计min/max/avg）

        Args:
            name: 指标名称
            value: 指标值
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricValue(value)
            else:
                self._metrics[name].update(value)

    def increment(self, name: str, delta: int = 1):
        """
        增加计数器

        Args:
            name: 计数器名称
            delta: 增量（默认1）
        """
        with self._lock:
            self._counters[name] += delta

    def increment_counter(self, name: str, delta: int = 1):
        """
        增加计数器（increment的别名，保持API一致性）

        Args:
            name: 计数器名称
            delta: 增量（默认1）
        """
        self.increment(name, delta)

    def set_gauge(self, name: str, value: float):
        """
        设置仪表值（瞬时值）

        Args:
            name: 仪表名称
            value: 仪表值
        """
        with self._lock:
            self._gauges[name] = value

    def get_metric(self, name: str) -> Optional[MetricValue]:
        """获取指标"""
        with self._lock:
            return self._metrics.get(name)

    def get_counter(self, name: str) -> int:
        """获取计数器"""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """获取仪表值"""
        with self._lock:
            return self._gauges.get(name)

    def get_all(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            result = {}

            # MetricValue类型指标
            for name, metric in self._metrics.items():
                result[name] = {
                    "value": metric.value,
                    "count": metric.count,
                    "min": metric.min if metric.min != float('inf') else 0.0,
                    "max": metric.max if metric.max != float('-inf') else 0.0,
                    "avg": metric.avg,
                }

            # 计数器
            for name, count in self._counters.items():
                result[name] = count

            # 仪表
            result.update(self._gauges)

            return result

    def reset(self):
        """重置所有指标"""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()


class SystemMetrics:
    """系统指标监控"""

    @staticmethod
    def get_cpu_percent() -> float:
        """获取CPU使用率"""
        return psutil.cpu_percent(interval=0.1)

    @staticmethod
    def get_memory_mb() -> float:
        """获取内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def get_thread_count() -> int:
        """获取线程数"""
        return psutil.Process().num_threads()


class LatencyTimer:
    """延迟计时器（上下文管理器）"""

    def __init__(self, collector: MetricsCollector, metric_name: str):
        """
        初始化计时器

        Args:
            collector: 指标收集器
            metric_name: 指标名称
        """
        self.collector = collector
        self.metric_name = metric_name
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.collector.record(self.metric_name, elapsed_ms)


# 全局指标收集器实例
_global_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """获取全局指标收集器实例"""
    global _global_collector

    if _global_collector is None:
        _global_collector = MetricsCollector()

    return _global_collector


def record_latency(name: str, value: float):
    """记录延迟指标"""
    get_metrics().record(name, value)


def increment_counter(name: str, delta: int = 1):
    """增加计数器"""
    get_metrics().increment(name, delta)


def set_gauge(name: str, value: float):
    """设置仪表值"""
    get_metrics().set_gauge(name, value)

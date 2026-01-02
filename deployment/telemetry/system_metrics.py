"""Raspberry Pi system metrics collection.

Collects CPU, memory, disk, and temperature metrics efficiently.
Designed for low overhead in real-time control context.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to import psutil, gracefully degrade if unavailable
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available, system metrics will be unavailable", file=sys.stderr)


# RPi thermal zone path
THERMAL_ZONE_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


@dataclass
class SystemMetrics:
    """System metrics snapshot.

    Attributes:
        cpu_temp_c: CPU temperature in Celsius
        cpu_percent: CPU usage percentage (0-100)
        memory_percent: Memory usage percentage (0-100)
        disk_percent: Disk usage percentage (0-100)
        valid: True if all metrics collected successfully
    """
    cpu_temp_c: float = -1.0
    cpu_percent: float = -1.0
    memory_percent: float = -1.0
    disk_percent: float = -1.0
    valid: bool = False


class SystemMetricsCollector:
    """Collects RPi system metrics with caching.

    Caches metrics for a configurable interval to reduce overhead.
    Uses psutil for cross-platform compatibility.

    Usage:
        collector = SystemMetricsCollector(cache_duration_s=0.5)
        metrics = collector.get()  # Returns cached or fresh metrics
    """

    def __init__(self, cache_duration_s: float = 0.5):
        """Initialize collector.

        Args:
            cache_duration_s: How long to cache metrics. At 5 Hz logging,
                             0.5s means fresh metrics every 2-3 log calls.
        """
        self._cache_duration = cache_duration_s
        self._cached_metrics: Optional[SystemMetrics] = None
        self._cache_time: float = 0.0

    def get(self) -> SystemMetrics:
        """Get current system metrics (may be cached).

        Never raises exceptions. Returns SystemMetrics with valid=False
        if collection fails.
        """
        now = time.time()

        # Return cached if still fresh
        if self._cached_metrics is not None and (now - self._cache_time) < self._cache_duration:
            return self._cached_metrics

        # Collect fresh metrics
        metrics = SystemMetrics()

        try:
            # CPU temperature (RPi-specific)
            cpu_temp = self._get_cpu_temperature()
            if cpu_temp is not None:
                metrics.cpu_temp_c = cpu_temp

            # psutil metrics
            if PSUTIL_AVAILABLE:
                metrics.cpu_percent = psutil.cpu_percent(interval=None)
                metrics.memory_percent = psutil.virtual_memory().percent
                metrics.disk_percent = psutil.disk_usage("/").percent

            metrics.valid = True

        except Exception as e:
            print(f"WARNING: System metrics collection failed: {e}", file=sys.stderr)

        # Cache the result
        self._cached_metrics = metrics
        self._cache_time = now

        return metrics

    @staticmethod
    def _get_cpu_temperature() -> Optional[float]:
        """Read CPU temperature from sysfs.

        RPi-specific: reads from /sys/class/thermal/thermal_zone0/temp
        Returns None if unavailable.
        """
        try:
            if THERMAL_ZONE_PATH.exists():
                with open(THERMAL_ZONE_PATH) as f:
                    # Value is in millidegrees Celsius
                    return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            pass
        return None


if __name__ == "__main__":
    # Test system metrics collection
    collector = SystemMetricsCollector(cache_duration_s=0.5)

    print("Testing system metrics collection...")
    for i in range(5):
        metrics = collector.get()
        print(f"  [{i}] CPU: {metrics.cpu_temp_c:.1f}C, {metrics.cpu_percent:.1f}% | "
              f"Mem: {metrics.memory_percent:.1f}% | Disk: {metrics.disk_percent:.1f}% | "
              f"Valid: {metrics.valid}")
        time.sleep(0.3)

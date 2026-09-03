# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Fixed-memory metric primitives used by FastOffload telemetry."""

import math
import threading
from dataclasses import dataclass
from numbers import Real
from typing import Dict, Optional, Union

MetricValue = Union[int, float]


def _validate_metric_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Metric name must be a non-empty string")


def _validate_metric_value(value: MetricValue) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Metric value must be an int or float")
    if not math.isfinite(value):
        raise ValueError("Metric value must be finite")


@dataclass(frozen=True)
class RunningStatsSnapshot:
    """Immutable summary of observations collected for one metric."""

    count: int
    total: float
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]


class RunningStats:
    """Track count, total, minimum, maximum, and mean in constant memory."""

    def __init__(self) -> None:
        self.reset()

    def observe(self, value: MetricValue) -> None:
        _validate_metric_value(value)
        numeric_value = float(value)
        self.count += 1
        self.total += numeric_value
        if self.minimum is None or numeric_value < self.minimum:
            self.minimum = numeric_value
        if self.maximum is None or numeric_value > self.maximum:
            self.maximum = numeric_value

    def snapshot(self) -> RunningStatsSnapshot:
        mean = self.total / self.count if self.count else None
        return RunningStatsSnapshot(count=self.count,
                                    total=self.total,
                                    minimum=self.minimum,
                                    maximum=self.maximum,
                                    mean=mean)

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None


@dataclass(frozen=True)
class MetricsSnapshot:
    """Point-in-time copy of all registered metric values."""

    counters: Dict[str, MetricValue]
    gauges: Dict[str, MetricValue]
    histograms: Dict[str, RunningStatsSnapshot]


class MetricsRegistry:
    """Thread-safe, fixed-memory aggregation for FastOffload metrics.

    Counters and histograms represent values collected during a reporting
    window. Gauges represent current state, so ``snapshot(reset=True)`` retains
    them while starting a new counter and histogram window.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, MetricValue] = {}
        self._gauges: Dict[str, MetricValue] = {}
        self._histograms: Dict[str, RunningStats] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: MetricValue = 1) -> None:
        """Increase a monotonic counter by a non-negative value."""
        _validate_metric_name(name)
        _validate_metric_value(value)
        if value < 0:
            raise ValueError("Counter increment must be non-negative")

        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: MetricValue) -> None:
        """Set the current value of a gauge."""
        _validate_metric_name(name)
        _validate_metric_value(value)

        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: MetricValue) -> None:
        """Add a sample to a fixed-memory running summary."""
        _validate_metric_name(name)
        _validate_metric_value(value)

        with self._lock:
            stats = self._histograms.get(name)
            if stats is None:
                stats = RunningStats()
                self._histograms[name] = stats
            stats.observe(value)

    def snapshot(self, reset: bool = False) -> MetricsSnapshot:
        """Return a consistent metric snapshot.

        When ``reset`` is true, counters and histograms start a new reporting
        window after the snapshot. Gauges are retained because they describe
        current state rather than values accumulated during a window.
        """
        with self._lock:
            snapshot = MetricsSnapshot(counters=dict(self._counters),
                                       gauges=dict(self._gauges),
                                       histograms={
                                           name: stats.snapshot()
                                           for name, stats in self._histograms.items()
                                       })
            if reset:
                self._counters.clear()
                self._histograms.clear()
            return snapshot

    def reset(self, reset_gauges: bool = True) -> None:
        """Clear collected metrics, optionally retaining current gauges."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            if reset_gauges:
                self._gauges.clear()

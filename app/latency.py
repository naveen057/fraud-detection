"""Thread-safe rolling latency tracker (p50 / p95 / p99 + SLA violations)."""
import threading
from collections import deque

import numpy as np


class LatencyTracker:
    def __init__(self, window: int = 5000, sla_ms: float = 50.0):
        self.sla_ms = sla_ms
        self._lat = deque(maxlen=window)
        self._lock = threading.Lock()
        self.total_requests = 0
        self.sla_violations = 0

    def record(self, ms: float) -> None:
        with self._lock:
            self._lat.append(ms)
            self.total_requests += 1
            if ms > self.sla_ms:
                self.sla_violations += 1

    def snapshot(self) -> dict:
        with self._lock:
            arr = np.array(self._lat, dtype=float)
            total, viol = self.total_requests, self.sla_violations
        if arr.size == 0:
            return {"total_requests": total, "window_size": 0, "sla_ms": self.sla_ms}
        return {
            "total_requests": total,
            "window_size": int(arr.size),
            "mean_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "max_ms": round(float(arr.max()), 3),
            "sla_ms": self.sla_ms,
            "sla_violations": viol,
            "sla_violation_rate": round(viol / max(total, 1), 5),
        }

# harness/latency_tracker.py
import statistics


class LatencyTracker:
    """Accumulates call durations and reports percentiles."""

    def __init__(self):
        self._samples: list[float] = []

    def record(self, total_ms: float) -> None:
        self._samples.append(total_ms)

    def p95(self) -> float | None:
        if len(self._samples) < 20:
            return None  # not enough data for a meaningful percentile
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx]

    def report(self) -> dict:
        if not self._samples:
            return {}
        return {
            "count": len(self._samples),
            "mean_ms": round(statistics.mean(self._samples)),
            "p50_ms": round(statistics.median(self._samples)),
            "p95_ms": round(self.p95() or 0),
            "max_ms": round(max(self._samples)),
        }

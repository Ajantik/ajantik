"""Honest summaries from few measurements.

Success rate: Wilson interval (80%), plus the rule of three when no failure was seen.
Cost per run: empirical p10/p50/p90, widened around the median in log space when there are few
runs, k = 1 + c/(n-1), c = 2.9, k <= 3 (agent-preflight experiment 003: fitted on 79 Exgentic
setups, held on unseen OpenHands data). The mean is never widened.
Total for many runs: double bootstrap (resample the pool = what we know, then draw the runs =
run-to-run variation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z80 = 1.2816  # two-sided 80% normal quantile
SMALL_SAMPLE_C = 2.9
MAX_WIDENING = 3.0


@dataclass
class RateSummary:
    n: int
    successes: int
    low: float | None
    high: float | None
    max_failure_rate_95: float | None  # rule of three, only when no failure was seen

    @property
    def rate(self) -> float | None:
        return self.successes / self.n if self.n else None


def wilson(successes: int, n: int, z: float = Z80) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize_rate(outcomes: list[bool]) -> RateSummary:
    n, k = len(outcomes), sum(outcomes)
    if n == 0:
        return RateSummary(0, 0, None, None, None)
    low, high = wilson(k, n)
    return RateSummary(n, k, low, high, min(1.0, 3 / n) if k == n else None)


def widening(n: int, c: float = SMALL_SAMPLE_C) -> float:
    if n < 2:
        return 1.0
    return min(1.0 + c / (n - 1), MAX_WIDENING)


def widen(values: np.ndarray, k: float) -> np.ndarray:
    """Stretch each value's distance from the median by k in log space."""
    median = float(np.median(values))
    if k == 1 or median <= 0:
        return values
    return median * np.power(np.maximum(values, 1e-12) / median, k)


@dataclass
class CostSummary:
    n: int
    mean: float | None
    p10: float | None
    p50: float | None
    p90: float | None
    widening: float
    total_runs: int
    total_p10: float | None
    total_p50: float | None
    total_p90: float | None


def summarize_cost(
    costs: list[float], total_runs: int = 1000, n_sim: int = 4000, seed: int = 11
) -> CostSummary:
    n = len(costs)
    if n == 0:
        return CostSummary(0, None, None, None, None, 1.0, total_runs, None, None, None)
    x = np.asarray(costs, dtype=float)
    mean = float(x.mean())
    if n < 2:  # one measurement: no spread to report
        return CostSummary(n, mean, None, float(x[0]), None, 1.0, total_runs, None, None, None)
    k = widening(n)
    p10, p50, p90 = np.percentile(widen(x, k), [10, 50, 90])
    rng = np.random.default_rng(seed)
    worlds = rng.choice(x, size=(n_sim, n), replace=True)
    world_means = worlds.mean(axis=1)
    # Sum of total_runs draws from each world ~ total_runs * world mean + CLT noise.
    world_sd = worlds.std(axis=1, ddof=0)
    totals = total_runs * world_means + rng.standard_normal(n_sim) * world_sd * math.sqrt(total_runs)
    t10, t50, t90 = np.percentile(totals, [10, 50, 90])
    return CostSummary(
        n, mean, float(p10), float(p50), float(p90), k, total_runs, float(t10), float(t50), float(t90)
    )

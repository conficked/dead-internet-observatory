"""
Gold-layer analytics — time-series computations over the composite index.

Generates:
  - Rolling decay rate (slope of smoothed IAI over configurable windows)
  - Source divergence score (std across source scores on same day)
  - A 100-minus-index helper that is not a calibrated authorship share
  - Trend classification (stable / declining / recovering / crash)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def compute_decay_rate(series: pd.Series, window: int = 90) -> pd.Series:
    """Rolling linear regression slope (points/day) over `window` days."""
    slopes = []
    x = np.arange(window, dtype=float)
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(np.nan)
        else:
            y = series.iloc[i - window + 1 : i + 1].values.astype(float)
            if np.isnan(y).any():
                slopes.append(np.nan)
            else:
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(round(float(slope), 4))
    return pd.Series(slopes, index=series.index)


def classify_trend(slope_per_day: float, threshold: float = 0.05) -> str:
    if abs(slope_per_day) < threshold:
        return "stable"
    elif slope_per_day < -0.3:
        return "crash"
    elif slope_per_day < 0:
        return "declining"
    elif slope_per_day > 0:
        return "recovering"
    return "stable"


def synthetic_content_estimate(iai: float) -> Tuple[float, float, float]:
    """
    Return (point, lower, upper) for 100 minus the index.

    This helper is not a calibrated share of synthetic text, and the dashboard
    does not publish it as one. The band is a fixed formula that widens near
    the middle of the 0–100 scale. It is not a sampling interval.
    """
    point = 100.0 - iai
    std = 5.0 + 10.0 * (1.0 - abs(iai - 50) / 50.0)   # max uncertainty at 50
    return (
        round(point, 1),
        round(max(0, point - 1.96 * std), 1),
        round(min(100, point + 1.96 * std), 1),
    )


def enrich_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived analytics columns to a composite_index DataFrame."""
    df = df.copy().sort_values("date")

    df["decay_rate_90d"] = compute_decay_rate(df["aliveness_index"], 90)
    df["decay_rate_30d"] = compute_decay_rate(df["aliveness_index"], 30)

    df["trend"] = df["decay_rate_30d"].apply(
        lambda s: classify_trend(s) if not pd.isna(s) else "unknown"
    )

    synth = df["aliveness_index"].apply(synthetic_content_estimate)
    df["synthetic_pct"]      = synth.apply(lambda t: t[0])
    df["synthetic_pct_low"]  = synth.apply(lambda t: t[1])
    df["synthetic_pct_high"] = synth.apply(lambda t: t[2])

    return df

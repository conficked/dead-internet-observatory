"""Source-level behavioral summaries.

Burstiness and inter-event regularity stay here. They are not document
sub-scores and they do not enter the headline composite.
"""

import numpy as np
import pandas as pd

from detection.bot_patterns import detect_regular_intervals
from detection.linguistic_features import compute_burstiness


BEHAVIOR_COLUMNS = [
    "source", "n_events", "burstiness", "regularity",
    "interevent_mean_seconds", "interevent_std_seconds",
]
MOVEMENT_COLUMNS = [
    "date", "previous_date", "index", "previous_index",
    "score_effect", "composition_effect", "entered", "exited",
]


def _seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.astype("int64") / 1e9


def summarize_source_behavior(events: pd.DataFrame) -> pd.DataFrame:
    """One row per source. The frame has no aliveness_score column."""
    if events.empty or "source" not in events or "timestamp" not in events:
        return pd.DataFrame(columns=BEHAVIOR_COLUMNS)
    frame = events.copy()
    frame["seconds"] = _seconds(frame["timestamp"])
    rows = []
    for source, group in frame.groupby("source", sort=True):
        ordered = sorted(float(value) for value in group["seconds"].dropna())
        intervals = np.diff(ordered) if len(ordered) >= 2 else np.array([])
        positive = intervals[intervals > 0] if len(intervals) else intervals
        rows.append({
            "source": source,
            "n_events": int(len(ordered)),
            "burstiness": float(compute_burstiness(ordered)),
            "regularity": float(detect_regular_intervals(ordered)),
            "interevent_mean_seconds": float(positive.mean()) if len(positive) else None,
            "interevent_std_seconds": float(positive.std()) if len(positive) else None,
        })
    return pd.DataFrame(rows, columns=BEHAVIOR_COLUMNS)


def _weighted_index(means: dict, weights: dict):
    names = [name for name in means if name in weights and weights[name] > 0]
    if not names:
        return None
    values = np.array([float(means[name]) for name in names], dtype=float)
    used = np.array([float(weights[name]) for name in names], dtype=float)
    return float(np.average(values, weights=used))


def attribute_index_movement(frame: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Split a change in the fixed-weight mean into score and composition parts.

    score_effect holds the source mix at the intersection of the two days and
    changes only the source means. composition_effect is the rest of the change,
    from sources that entered or left. Weights are design weights. Within the
    sources present on a day they are renormalized the same way as a published
    headline that has already passed its coverage gate.
    """
    if frame.empty:
        return pd.DataFrame(columns=MOVEMENT_COLUMNS)
    work = frame.copy()
    work["date"] = work["date"].astype(str)
    by_date = {}
    for date, group in work.groupby("date", sort=True):
        by_date[str(date)] = {
            str(source): float(score)
            for source, score in zip(group["source"], group["mean_score"])
        }
    dates = sorted(by_date)
    rows = []
    for previous, current in zip(dates, dates[1:]):
        old, new = by_date[previous], by_date[current]
        shared_names = sorted(set(old) & set(new))
        shared_new = {name: new[name] for name in shared_names}
        shared_old = {name: old[name] for name in shared_names}
        current_index = _weighted_index(new, weights)
        previous_index = _weighted_index(old, weights)
        shared_current = _weighted_index(shared_new, weights)
        shared_previous = _weighted_index(shared_old, weights)
        if shared_current is None or shared_previous is None or current_index is None or previous_index is None:
            score_effect = None
            composition = None
        else:
            score_effect = shared_current - shared_previous
            composition = (current_index - previous_index) - score_effect
        rows.append({
            "date": current,
            "previous_date": previous,
            "index": current_index,
            "previous_index": previous_index,
            "score_effect": score_effect,
            "composition_effect": composition,
            "entered": sorted(set(new) - set(old)),
            "exited": sorted(set(old) - set(new)),
        })
    return pd.DataFrame(rows, columns=MOVEMENT_COLUMNS)

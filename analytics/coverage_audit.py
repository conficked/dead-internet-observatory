"""Compare the configured headline gate with the history that is actually on disk.

Biweekly platform bins and Common Crawl crawl rows are separate series. Crawl
median dates are not copied into the biweekly bins. A missing daily ledger is
reported as no daily rows, which is different from zero days losing the headline.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import yaml


def audit_headline_coverage(timeline_dir, db_path, config_path) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())
    analytics = config.get("analytics", {})
    weights = analytics.get("source_weights", {})
    panel = list(analytics.get("core_source_panel", weights))
    minimum_coverage = float(analytics.get("minimum_coverage", 1.0))
    minimum_source_samples = int(analytics.get("minimum_source_samples", 1))
    timeline = Path(timeline_dir)
    biweekly = pd.read_csv(timeline / "biweekly.csv")
    crawls = pd.read_csv(timeline / "cc_crawls.csv")
    periods = 0
    meeting = 0
    max_coverage = 0.0
    for _, group in biweekly.groupby("period_start"):
        periods += 1
        present = 0.0
        for source, count in zip(group["source"], group["n"]):
            if source in weights and int(count) >= minimum_source_samples:
                present += float(weights[source])
        max_coverage = max(max_coverage, present)
        if present >= minimum_coverage:
            meeting += 1
    minimum_total_samples = int(analytics.get("minimum_total_samples", 1))
    sqlite_dates = 0
    days_losing = None
    database = Path(db_path)
    if database.exists():
        uri = f"file:{database}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "daily_index" in tables:
                counted = conn.execute(
                    """SELECT date, source, SUM(n_docs) AS n_docs
                       FROM daily_index GROUP BY date, source"""
                ).fetchall()
                by_date = {}
                for date, source, count in counted:
                    by_date.setdefault(date, []).append((source, int(count or 0)))
                sqlite_dates = len(by_date)
                if sqlite_dates:
                    panel_weight = sum(float(weights.get(source, 0.0)) for source in panel) or 1.0
                    days_losing = 0
                    for entries in by_date.values():
                        present = sum(
                            float(weights.get(source, 0.0))
                            for source, count in entries
                            if source in panel and count >= minimum_source_samples
                        )
                        total = sum(count for source, count in entries if source in panel)
                        if present / panel_weight < minimum_coverage or total < minimum_total_samples:
                            days_losing += 1
    observed = set(biweekly["source"].astype(str))
    gold = Path(config_path).resolve().parents[1] / "gold" / "scored.parquet"
    return {
        "recommendation": "keep",
        "configured_minimum_coverage": minimum_coverage,
        "recommended_minimum_coverage": minimum_coverage,
        "minimum_source_samples": minimum_source_samples,
        "minimum_total_samples": minimum_total_samples,
        "biweekly_periods": int(periods),
        "biweekly_periods_meeting_coverage": int(meeting),
        "max_biweekly_panel_coverage": round(max_coverage, 6),
        "panel_sources_absent_from_biweekly": sorted(
            source for source in panel if source not in observed),
        "common_crawl_crawls": int(len(crawls)),
        "sqlite_daily_dates": int(sqlite_dates),
        "days_losing_headline": days_losing,
        "scored_parquet_present": gold.exists(),
        "note": (
            "Biweekly bins are not daily rows, and Common Crawl crawl medians "
            "are not assigned to those bins. An empty daily ledger means there "
            "is no count of days losing the headline. The coverage threshold is unchanged."
        ),
    }

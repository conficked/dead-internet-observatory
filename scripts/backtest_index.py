"""Replay scored documents in date order and check that trailing smoothing holds.

The input is JSONL with doc_id, source, category, created_dt, aliveness_score
and scoring_version. domain is optional and is the bootstrap cluster. The
script writes a temporary SQLite file and does not use a paid API.

    .venv/bin/python scripts/backtest_index.py \
        --input tests/fixtures/scored_documents.jsonl \
        --config tests/fixtures/backtest_config.yaml \
        --output /tmp/backtest.json \
        --seed 0
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.aliveness_index import AlivenessIndexEngine
from analytics.analysis_io import SCHEMA_VERSION, file_sha256, write_json


def _number(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _same(left, right) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) < 1e-6


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Scored JSONL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    loaded = yaml.safe_load(Path(args.config).read_text())
    frame = pd.read_json(args.input, lines=True)
    frame["created_dt"] = pd.to_datetime(frame["created_dt"], utc=True, errors="coerce")
    with tempfile.TemporaryDirectory() as directory:
        loaded.setdefault("storage", {})["db_path"] = str(Path(directory) / "backtest.db")
        loaded.setdefault("analytics", {})["bootstrap_seed"] = int(args.seed)
        config_path = Path(directory) / "config.yaml"
        config_path.write_text(yaml.safe_dump(loaded))
        engine = AlivenessIndexEngine(str(config_path))
        history = {}
        stable = True
        for date in sorted(frame["created_dt"].dt.strftime("%Y-%m-%d").dropna().unique()):
            batch = frame[frame["created_dt"].dt.strftime("%Y-%m-%d").eq(date)]
            engine.ingest_scored_delta(batch)
            timeline = engine.get_composite_timeline(days=100000)
            for previous, value in history.items():
                current = timeline.loc[timeline["date"].astype(str).eq(previous), "smoothed_index"]
                observed = None if current.empty or pd.isna(current.iloc[0]) else float(current.iloc[0])
                if not _same(observed, value):
                    stable = False
            history = {
                str(row.date): (None if pd.isna(row.smoothed_index) else float(row.smoothed_index))
                for row in timeline.itertuples()
            }
        timeline = engine.get_composite_timeline(days=100000)
        analytics = loaded.get("analytics", {})
        days = []
        for row in timeline.itertuples():
            published = (
                row.aliveness_index is not None and not pd.isna(row.aliveness_index)
                and int(row.coverage_sufficient or 0) == 1
                and int(row.sample_sufficient or 0) == 1
            )
            days.append({
                "date": str(row.date),
                "aliveness_index": _number(row.aliveness_index),
                "smoothed_index": _number(row.smoothed_index),
                "source_coverage": _number(row.source_coverage),
                "coverage_sufficient": int(row.coverage_sufficient or 0),
                "sample_sufficient": int(row.sample_sufficient or 0),
                "headline_published": bool(published),
                "ci_low": _number(row.ci_low),
                "ci_high": _number(row.ci_high),
            })
    write_json(args.output, {
        "script": "backtest_index",
        "schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "input_sha256": file_sha256(args.input),
        "thresholds": {
            "minimum_coverage": analytics.get("minimum_coverage"),
            "minimum_source_samples": analytics.get("minimum_source_samples"),
            "minimum_total_samples": analytics.get("minimum_total_samples"),
            "index_smoothing_window": analytics.get("index_smoothing_window"),
        },
        "days": days,
        "future_leakage_check": {"history_stable": stable},
        "limitations": [
            "The replay uses scores already stored in the input. It does not rescore text.",
            "A stable history check confirms trailing smoothing on this file. It is not a forecast test.",
        ],
    })


if __name__ == "__main__":
    main()

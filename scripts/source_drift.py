"""Separate source-score movement from source-composition change.

The input CSV needs date, source, mean_score and n_docs. Sources in the file
receive equal design weights unless the rows already describe a weighted panel.
This script does not read document text.

    .venv/bin/python scripts/source_drift.py \
        --input tests/fixtures/source_daily.csv \
        --output /tmp/drift.json \
        --seed 0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.analysis_io import SCHEMA_VERSION, file_sha256, write_json
from analytics.source_behavior import attribute_index_movement


def _number(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with date, source, mean_score, n_docs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.input)
    required = {"date", "source", "mean_score", "n_docs"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"source drift input requires columns: {', '.join(sorted(missing))}")
    weights = {str(source): 1.0 for source in sorted(frame["source"].astype(str).unique())}
    movement = attribute_index_movement(frame, weights)
    sources = []
    for source, group in frame.groupby("source", sort=True):
        ordered = group.sort_values("date")
        first = float(ordered.iloc[0]["mean_score"])
        last = float(ordered.iloc[-1]["mean_score"])
        sources.append({
            "source": str(source),
            "n_days": int(ordered["date"].nunique()),
            "first_mean": round(first, 6),
            "last_mean": round(last, 6),
            "delta": round(last - first, 6),
        })
    rows = []
    for row in movement.itertuples(index=False):
        rows.append({
            "date": row.date,
            "previous_date": row.previous_date,
            "score_effect": _number(row.score_effect),
            "composition_effect": _number(row.composition_effect),
            "entered": list(row.entered),
            "exited": list(row.exited),
        })
    write_json(args.output, {
        "script": "source_drift",
        "schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "input_sha256": file_sha256(args.input),
        "weighting": "equal weights over sources present in the input",
        "sources": sources,
        "movement": rows,
        "limitations": [
            "Equal weights describe the supplied table. They are not the live panel weights.",
            "Composition change is sources entering or leaving the table, not a change in collection effort that is invisible in the rows.",
        ],
    })


if __name__ == "__main__":
    main()

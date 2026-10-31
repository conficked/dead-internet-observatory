"""Fit a regularized logistic baseline on labeled human/generated documents.

The input is JSONL or CSV with the common schema documented in
analytics/calibration.py. The output is JSON. Scaling in that JSON can replace
detection.feature_scaling. Headline weights are left unchanged below 200 rows.

    .venv/bin/python scripts/calibrate_detector.py \
        --input tests/fixtures/labeled_documents.jsonl \
        --config config/config.yaml \
        --output /tmp/calibration.json \
        --seed 0
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.analysis_io import file_sha256, write_json
from analytics.calibration import build_calibration_report, load_labeled_dataset


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled JSONL or CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    frame = load_labeled_dataset(args.input)
    report = build_calibration_report(
        frame,
        seed=args.seed,
        weights=config.get("detection", {}).get("weights", {}),
        input_sha256=file_sha256(args.input),
    )
    write_json(args.output, report)


if __name__ == "__main__":
    main()

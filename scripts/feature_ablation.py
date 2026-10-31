"""Measure feature correlation and leave-one-feature-out fit on labeled text.

Vocabulary-richness features are reported together. The provisional headline
keeps MTLD and excludes the rest of that family.

    .venv/bin/python scripts/feature_ablation.py \
        --input tests/fixtures/labeled_documents.jsonl \
        --output /tmp/ablation.json \
        --seed 0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.analysis_io import file_sha256, write_json
from analytics.calibration import build_ablation_report, load_labeled_dataset


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled JSONL or CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    frame = load_labeled_dataset(args.input)
    report = build_ablation_report(frame, seed=args.seed, input_sha256=file_sha256(args.input))
    write_json(args.output, report)


if __name__ == "__main__":
    main()

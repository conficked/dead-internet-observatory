"""Perturb headline weights with a fixed-seed Dirichlet and record the score spread.

    .venv/bin/python scripts/weight_sensitivity.py \
        --input tests/fixtures/labeled_documents.jsonl \
        --config config/config.yaml \
        --output /tmp/sensitivity.json \
        --seed 0
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.analysis_io import SCHEMA_VERSION, file_sha256, write_json
from analytics.calibration import feature_tables, load_labeled_dataset, weight_sensitivity


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled JSONL or CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=100)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    weights = config.get("detection", {}).get("weights", {})
    frame = load_labeled_dataset(args.input)
    _raw, scores = feature_tables(frame)
    spread = weight_sensitivity(scores, weights, seed=args.seed, n_draws=args.draws)
    write_json(args.output, {
        "script": "weight_sensitivity",
        "schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "input_sha256": file_sha256(args.input),
        **spread,
        "limitations": [
            "Draws are Dirichlet perturbations around the configured weights, concentration 80.",
            "The spread describes this input file. It is not a confidence interval for the live index.",
        ],
    })


if __name__ == "__main__":
    main()

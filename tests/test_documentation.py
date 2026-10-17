from pathlib import Path


def test_no_fabricated_historical_baselines():
    root = Path(__file__).resolve().parents[1]
    for name in ['README.md', 'config/config.yaml', 'config/config_fast.yaml']:
        text = (root / name).read_text()
        assert 'baseline_aliveness' not in text
        assert '2019 baseline' not in text
        assert 'historical_baseline_year' not in text

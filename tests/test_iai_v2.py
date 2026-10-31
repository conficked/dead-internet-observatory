import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from analytics.aliveness_index import AlivenessIndexEngine
from detection.ai_content_detector import score_dataframe, score_document, validate_weights
from detection.linguistic_features import extract_features

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def write_config(tmp_path, *, source_weights=None, core_sources=None,
                 minimum_source_samples=1, minimum_total_samples=1,
                 minimum_coverage=1.0, smoothing_window=3):
    source_weights = source_weights or {"reddit": 0.5, "news": 0.5}
    core_sources = core_sources or list(source_weights)
    config = {
        "storage": {"db_path": str(tmp_path / "index.db")},
        "detection": {
            "scoring_version": "iai-v2-test",
            "weights": {
                "mtld": 0.15,
                "sentence_variance": 0.15,
                "repetition": 0.15,
                "zipf_deviation": 0.10,
                "compression_ratio": 0.10,
                "character_entropy": 0.10,
                "hapax_ratio": 0.10,
                "yules_k": 0.05,
                "lexical_diversity": 0.10,
            },
        },
        "analytics": {
            "source_weights": source_weights,
            "core_source_panel": core_sources,
            "minimum_source_samples": minimum_source_samples,
            "minimum_total_samples": minimum_total_samples,
            "minimum_coverage": minimum_coverage,
            "index_smoothing_window": smoothing_window,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def scored_docs(date, source, category, scores, version="iai-v2-test"):
    return pd.DataFrame({
        "doc_id": [f"{date}-{source}-{category}-{i}" for i in range(len(scores))],
        "source": source,
        "category": category,
        "created_dt": date,
        "aliveness_score": scores,
        "scoring_version": version,
    })


def test_weight_validation_rejects_invalid_sum_and_unknown_features():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_weights({"mtld": 0.9})
    with pytest.raises(ValueError, match="unknown"):
        validate_weights({"mtld": 0.5, "not_a_feature": 0.5})


def test_document_scoring_persists_version_and_excludes_temporal_and_perplexity_signals(tmp_path):
    config = write_config(tmp_path)
    result = score_document("One short sentence. A much longer sentence contains varied words and rhythms.",
                            timestamps=[1.0, 2.0, 4.0],
                            config_path=str(config))
    assert result["scoring_version"] == "iai-v2-test"
    assert "feat_burstiness" not in result
    assert "score_burstiness" not in result
    assert "score_perplexity" not in result


def _tiled_passage(repeats):
    from detection.linguistic_features import COMPRESSION_WINDOW_BYTES, TOKEN_WINDOW, tokenise
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words = [alphabet[i // 26] + alphabet[i % 26] for i in range(TOKEN_WINDOW - 2)]
    words.extend(words[:2])
    body = " ".join(words)
    deficit = COMPRESSION_WINDOW_BYTES - len(body.encode("utf-8"))
    assert deficit > 0
    unit = body + (" " * deficit)
    assert len(unit.encode("utf-8")) == COMPRESSION_WINDOW_BYTES
    assert len(tokenise(unit)) == TOKEN_WINDOW
    return unit * repeats


def test_cpu_features_are_finite_and_stable_across_document_lengths():
    unit = "alpha beta gamma delta alpha epsilon zeta eta theta alpha iota kappa "
    short = extract_features(unit * 5)
    long = extract_features(unit * 10)
    assert short["lexical_diversity"] == pytest.approx(long["lexical_diversity"])

    one = extract_features(_tiled_passage(1))
    many = extract_features(_tiled_passage(4))
    for name in ("compression_ratio", "character_entropy", "hapax_ratio", "yules_k",
                 "lexical_diversity"):
        assert pd.notna(one[name])
        assert one[name] == pytest.approx(many[name], abs=1e-9)
    assert one["compression_ratio_raw"] != pytest.approx(many["compression_ratio_raw"], abs=1e-6)
    assert one["hapax_ratio_raw"] != pytest.approx(many["hapax_ratio_raw"], abs=1e-6)


def test_score_dataframe_uses_one_configured_regime_for_every_row(tmp_path):
    config = write_config(tmp_path)
    frame = pd.DataFrame({"text": ["alpha beta gamma", "delta epsilon zeta"]})
    scored = score_dataframe(frame, config_path=str(config))
    assert scored["scoring_version"].tolist() == ["iai-v2-test", "iai-v2-test"]


def test_source_is_aggregated_once_before_weighting_categories(tmp_path):
    engine = AlivenessIndexEngine(str(write_config(tmp_path)))
    frame = pd.concat([
        scored_docs("2026-01-01", "reddit", "a", [0, 0]),
        scored_docs("2026-01-01", "reddit", "b", [100, 100]),
        scored_docs("2026-01-01", "news", "news", [100, 100]),
    ], ignore_index=True)
    engine.ingest_scored_df(frame)
    row = engine.get_composite_timeline(days=10000).iloc[0]
    assert row.aliveness_index == 75.0
    assert row.source_coverage == 1.0


def test_missing_core_source_suppresses_headline_without_renormalizing(tmp_path):
    engine = AlivenessIndexEngine(str(write_config(tmp_path, minimum_coverage=0.75)))
    engine.ingest_scored_df(scored_docs("2026-01-01", "reddit", "social", [80, 80]))
    row = engine.get_composite_timeline(days=10000).iloc[0]
    assert row.source_coverage == 0.5
    assert pd.isna(row.aliveness_index)
    assert row.coverage_sufficient == 0


def test_minimum_source_and_total_samples_suppress_headline(tmp_path):
    config = write_config(tmp_path, minimum_source_samples=2, minimum_total_samples=5)
    engine = AlivenessIndexEngine(str(config))
    frame = pd.concat([
        scored_docs("2026-01-01", "reddit", "social", [80, 80]),
        scored_docs("2026-01-01", "news", "news", [60, 60]),
    ], ignore_index=True)
    engine.ingest_scored_df(frame)
    row = engine.get_composite_timeline(days=10000).iloc[0]
    assert row.source_coverage == 1.0
    assert row.sample_sufficient == 0
    assert pd.isna(row.aliveness_index)


def test_trailing_smoothing_does_not_change_history_when_future_day_arrives(tmp_path):
    config = write_config(tmp_path, source_weights={"reddit": 1.0},
                          core_sources=["reddit"], smoothing_window=3)
    engine = AlivenessIndexEngine(str(config))
    first = pd.concat([
        scored_docs("2026-01-01", "reddit", "social", [20]),
        scored_docs("2026-01-02", "reddit", "social", [40]),
    ], ignore_index=True)
    engine.ingest_scored_df(first)
    before = engine.get_composite_timeline(days=10000).set_index("date")["smoothed_index"].copy()
    engine.ingest_scored_delta(scored_docs("2026-01-03", "reddit", "social", [100]))
    after = engine.get_composite_timeline(days=10000).set_index("date")["smoothed_index"]
    pd.testing.assert_series_equal(after.loc[before.index], before)
    assert after.iloc[-1] == pytest.approx((20 + 40 + 100) / 3, abs=0.001)


def test_scoring_version_is_persisted_with_documents_and_aggregates(tmp_path):
    engine = AlivenessIndexEngine(str(write_config(tmp_path, source_weights={"reddit": 1.0},
                                                   core_sources=["reddit"])))
    engine.ingest_scored_df(scored_docs("2026-01-01", "reddit", "social", [70, 80]))
    with engine._conn() as conn:
        versions = conn.execute(
            "SELECT entity_type, scoring_version FROM scoring_regimes ORDER BY entity_type"
        ).fetchall()
    assert {(row[0], row[1]) for row in versions} == {
        ("composite", "iai-v2-test"),
        ("daily", "iai-v2-test"),
        ("document", "iai-v2-test"),
    }


def test_mixed_regimes_are_excluded_from_the_headline(tmp_path):
    engine = AlivenessIndexEngine(str(write_config(tmp_path)))
    frame = pd.concat([
        scored_docs("2026-01-01", "reddit", "a", [0, 0], version="iai-v2-test"),
        scored_docs("2026-01-01", "reddit", "b", [100, 100], version="legacy-v1"),
        scored_docs("2026-01-01", "news", "news", [100, 100], version="iai-v2-test"),
    ], ignore_index=True)
    engine.ingest_scored_df(frame)
    row = engine.get_composite_timeline(days=10000).iloc[0]
    assert row.aliveness_index == pytest.approx(50.0)
    assert row.scoring_version == "iai-v2-test"
    assert not str(row.scoring_version).startswith("mixed")
    assert int(row.noncurrent_docs) == 2


def test_legacy_only_source_does_not_fill_coverage(tmp_path):
    engine = AlivenessIndexEngine(str(write_config(tmp_path, minimum_coverage=0.75)))
    frame = pd.concat([
        scored_docs("2026-01-01", "reddit", "social", [20, 20], version="iai-v2-test"),
        scored_docs("2026-01-01", "news", "news", [90, 90], version="legacy-v1"),
    ], ignore_index=True)
    engine.ingest_scored_df(frame)
    row = engine.get_composite_timeline(days=10000).iloc[0]
    assert row.source_coverage == pytest.approx(0.5)
    assert pd.isna(row.aliveness_index)
    assert int(row.regime_sufficient) == 0


def test_unversioned_daily_rows_stay_out_of_the_current_headline(tmp_path):
    engine = AlivenessIndexEngine(str(write_config(tmp_path)))
    engine.ingest_scored_df(scored_docs("2026-01-01", "reddit", "social", [0, 0]))
    with engine._conn() as conn:
        conn.execute(
            "INSERT INTO daily_index VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-01-01", "news", "news", 4, 100, 100, 0, 0, 0, 100),
        )
    engine._recompute_composite()
    row = engine.get_composite_timeline(days=10000).iloc[0]
    assert row.source_coverage == pytest.approx(0.5)
    assert pd.isna(row.aliveness_index)


def test_provisional_scaling_is_labeled_and_replaceable():
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    fast = yaml.safe_load((ROOT / "config" / "config_fast.yaml").read_text())
    for loaded in (config, fast):
        weights = loaded["detection"]["weights"]
        assert sum(weights.values()) == pytest.approx(1.0)
        for name in ("hapax_ratio", "yules_k", "lexical_diversity", "ttr"):
            assert name not in weights
        assert loaded["detection"]["feature_scaling"]["provisional"] is True
    from detection.linguistic_features import features_to_scores
    features = extract_features(_tiled_passage(1))
    base = features_to_scores(features)
    replaced = features_to_scores(features, scaling={"hapax_ratio": {"divisor": 2.0}})
    assert replaced["hapax_ratio"] < base["hapax_ratio"]


def test_cluster_bootstrap_is_deterministic():
    from analytics.bootstrap import cluster_bootstrap_mean
    sums = np.array([0.0, 100.0])
    counts = np.array([2.0, 2.0])
    first = cluster_bootstrap_mean(sums, counts, seed=0, resamples=200)
    second = cluster_bootstrap_mean(sums, counts, seed=0, resamples=200)
    assert first == second
    # Two equal clusters only produce document-weighted means in {0, 25, 50}.
    # The percentile endpoints sit on that support, so another seed does not move them.
    # The weighted mean of these clusters is 25.
    assert first[0] < 25 < first[1]
    assert first[1] <= 50
    varied = np.arange(0.0, 110.0, 10.0)
    varied_counts = np.ones_like(varied)
    left = cluster_bootstrap_mean(varied, varied_counts, seed=0, resamples=200)
    right = cluster_bootstrap_mean(varied, varied_counts, seed=1, resamples=200)
    assert left == cluster_bootstrap_mean(varied, varied_counts, seed=0, resamples=200)
    assert left != right
    assert left[0] < 50 < left[1]
    assert cluster_bootstrap_mean(np.array([10.0]), np.array([2.0]), seed=0, resamples=20) == (None, None)


def test_source_and_composite_intervals_are_deterministic(tmp_path):
    config = write_config(
        tmp_path, source_weights={"reddit": 0.5, "news": 0.5},
        minimum_source_samples=1, minimum_total_samples=1, minimum_coverage=1.0,
    )
    engine = AlivenessIndexEngine(str(config))
    reddit = scored_docs("2026-01-01", "reddit", "social", [0, 0, 100, 100])
    reddit["domain"] = ["a.example", "a.example", "b.example", "b.example"]
    news = scored_docs("2026-01-01", "news", "news", [40, 40, 60, 60])
    news["domain"] = ["c.example", "c.example", "d.example", "d.example"]
    engine.ingest_scored_df(pd.concat([reddit, news], ignore_index=True))
    before = engine.get_composite_timeline(days=10000).iloc[0]
    engine._recompute_composite()
    after = engine.get_composite_timeline(days=10000).iloc[0]
    assert before.ci_low == after.ci_low
    assert before.ci_high == after.ci_high
    assert before.ci_low < before.aliveness_index < before.ci_high
    intervals = engine.get_source_intervals()
    assert set(intervals["source"]) == {"reddit", "news"}
    assert intervals["n_clusters"].min() == 2


def test_single_cluster_source_does_not_invent_an_interval(tmp_path):
    config = write_config(
        tmp_path, source_weights={"reddit": 1.0}, core_sources=["reddit"],
        minimum_source_samples=1, minimum_total_samples=1,
    )
    engine = AlivenessIndexEngine(str(config))
    frame = scored_docs("2026-01-01", "reddit", "social", [10, 90])
    frame["domain"] = "only.example"
    engine.ingest_scored_df(frame)
    row = engine.get_composite_timeline(days=10000).iloc[0]
    assert row.aliveness_index == pytest.approx(50)
    assert pd.isna(row.ci_low)
    assert pd.isna(row.ci_high)


def test_burstiness_stays_in_the_source_behavior_layer():
    from analytics.source_behavior import attribute_index_movement, summarize_source_behavior
    events = pd.DataFrame({
        "source": ["reddit"] * 5 + ["news"] * 4,
        "timestamp": [0, 10, 20, 30, 40, 0, 1, 2, 30],
    })
    summary = summarize_source_behavior(events).set_index("source")
    assert "aliveness_score" not in summary.columns
    assert summary.loc["reddit", "burstiness"] == pytest.approx(0.0)
    assert summary.loc["news", "n_events"] == 4
    movement = attribute_index_movement(pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"],
        "source": ["reddit", "news", "reddit", "news", "reddit"],
        "mean_score": [40, 80, 60, 80, 60],
        "n_docs": [10, 10, 10, 10, 10],
    }), {"reddit": 0.5, "news": 0.5}).set_index("date")
    assert movement.loc["2026-01-02", "score_effect"] == pytest.approx(10)
    assert movement.loc["2026-01-02", "composition_effect"] == pytest.approx(0)
    assert movement.loc["2026-01-03", "score_effect"] == pytest.approx(0)
    assert movement.loc["2026-01-03", "composition_effect"] == pytest.approx(-10)


def test_labeled_schema_rejects_missing_fields(tmp_path):
    from analytics.calibration import load_labeled_dataset
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"doc_id": "x", "text": "hello there"}) + "\n")
    with pytest.raises(ValueError, match="label"):
        load_labeled_dataset(bad)


def test_legacy_sqlite_and_parquet_stay_readable(tmp_path):
    from pipeline.silver_processing import append_scored
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE composite_index (
                date TEXT PRIMARY KEY, aliveness_index REAL, smoothed_index REAL,
                n_docs INTEGER, anomaly_flag INTEGER, anomaly_reason TEXT)"""
        )
        conn.execute(
            """CREATE TABLE daily_index (
                date TEXT, source TEXT, category TEXT, n_docs INTEGER, mean_score REAL,
                median_score REAL, std_score REAL, pct_below_50 REAL, bot_fraction REAL,
                aliveness_index REAL, PRIMARY KEY (date, source, category))"""
        )
        conn.execute(
            """CREATE TABLE ingested_docs (
                doc_id TEXT PRIMARY KEY, date TEXT, source TEXT, category TEXT,
                aliveness_score REAL, domain TEXT)"""
        )
        conn.execute(
            """CREATE TABLE domain_scores (
                date TEXT, domain TEXT, category TEXT, mean_score REAL, n_docs INTEGER,
                PRIMARY KEY (date, domain))"""
        )
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO composite_index VALUES ('2020-01-01', 61, 60, 20, 0, '')")
        conn.execute(
            "INSERT INTO daily_index VALUES ('2020-01-01','reddit','social',20,61,61,1,10,0,61)"
        )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"storage": {"db_path": str(db)}, "analytics": {}}))
    engine = AlivenessIndexEngine(str(config))
    timeline = engine.get_composite_timeline(days=20000)
    assert float(timeline.iloc[0].aliveness_index) == 61
    with engine._conn() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(composite_index)")}
        assert columns == {
            "date", "aliveness_index", "smoothed_index", "n_docs",
            "anomaly_flag", "anomaly_reason",
        }
        assert conn.execute("SELECT COUNT(*) FROM composite_index").fetchone()[0] == 1
    gold = tmp_path / "scored.parquet"
    old = pd.DataFrame({
        "doc_id": ["old"], "source": ["reddit"], "category": ["social"],
        "created_dt": ["2020-01-01"], "aliveness_score": [61.0], "text": ["hello world"],
    })
    old.to_parquet(gold, index=False)
    delta = old.copy()
    delta["doc_id"] = "new"
    delta["feat_compression_ratio"] = 0.4
    append_scored(gold, delta)
    loaded = pd.read_parquet(gold).set_index("doc_id")
    assert set(loaded.index) == {"old", "new"}
    assert loaded.loc["old", "aliveness_score"] == 61
    assert pd.isna(loaded.loc["old", "feat_compression_ratio"])


def test_coverage_audit_does_not_lower_the_configured_threshold():
    from analytics.coverage_audit import audit_headline_coverage
    report = audit_headline_coverage(
        timeline_dir=ROOT / "data" / "timeline",
        db_path=ROOT / "data" / "observatory.db",
        config_path=ROOT / "config" / "config.yaml",
    )
    assert report["recommendation"] == "keep"
    assert report["recommended_minimum_coverage"] == pytest.approx(report["configured_minimum_coverage"])
    assert report["configured_minimum_coverage"] == pytest.approx(0.8)
    assert report["biweekly_periods"] == 10
    assert report["biweekly_periods_meeting_coverage"] == 0
    assert report["max_biweekly_panel_coverage"] < 0.8
    assert report["sqlite_daily_dates"] == 0
    assert report["days_losing_headline"] is None


def _run_script(name, arguments):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *arguments],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return completed


def test_analysis_scripts_match_their_json_contracts(tmp_path):
    from analytics.calibration import load_labeled_dataset, ood_flags
    labeled = load_labeled_dataset(FIXTURES / "labeled_documents.jsonl")
    assert {"doc_id", "text", "label", "source", "genre", "language", "domain", "generator"} <= set(labeled.columns)
    flagged = ood_flags(labeled, labeled.assign(language="xx"))
    assert flagged[0]["language_ood"] is True
    outputs = {}
    for folder in ("first", "second"):
        directory = tmp_path / folder
        directory.mkdir()
        commands = {
            "calibration.json": [
                "calibrate_detector.py",
                "--input", str(FIXTURES / "labeled_documents.jsonl"),
                "--config", str(ROOT / "config" / "config.yaml"),
                "--output", str(directory / "calibration.json"),
                "--seed", "0",
            ],
            "backtest.json": [
                "backtest_index.py",
                "--input", str(FIXTURES / "scored_documents.jsonl"),
                "--config", str(FIXTURES / "backtest_config.yaml"),
                "--output", str(directory / "backtest.json"),
                "--seed", "0",
            ],
            "ablation.json": [
                "feature_ablation.py",
                "--input", str(FIXTURES / "labeled_documents.jsonl"),
                "--output", str(directory / "ablation.json"),
                "--seed", "0",
            ],
            "sensitivity.json": [
                "weight_sensitivity.py",
                "--input", str(FIXTURES / "labeled_documents.jsonl"),
                "--config", str(ROOT / "config" / "config.yaml"),
                "--output", str(directory / "sensitivity.json"),
                "--seed", "0",
            ],
            "drift.json": [
                "source_drift.py",
                "--input", str(FIXTURES / "source_daily.csv"),
                "--output", str(directory / "drift.json"),
                "--seed", "0",
            ],
        }
        for filename, command in commands.items():
            _run_script(command[0], command[1:])
        outputs[folder] = {
            name: json.loads((directory / name).read_text()) for name in commands
        }
    assert outputs["first"] == outputs["second"]
    calibration = outputs["first"]["calibration.json"]
    for key in (
        "script", "schema_version", "seed", "input_sha256", "n_documents", "model",
        "metrics", "stratified", "leave_one_generator_out", "leave_one_domain_out",
        "feature_scaling", "recommended_weights", "training_support", "ood_rule",
        "limitations",
    ):
        assert key in calibration
    assert calibration["seed"] == 0
    assert calibration["feature_scaling"]["provisional"] is True
    assert calibration["feature_scaling"]["sufficient_sample"] is False
    assert {row["held_out"] for row in calibration["leave_one_generator_out"]} == set(labeled["generator"])
    assert {row["held_out"] for row in calibration["leave_one_domain_out"]} == set(labeled["domain"])
    for column in ("source", "genre", "language", "length_bin"):
        assert column in calibration["stratified"]
    from detection.linguistic_features import features_to_scores
    rescored = features_to_scores(extract_features(_tiled_passage(1)), scaling=calibration["feature_scaling"])
    assert 0.0 <= rescored["compression_ratio"] <= 1.0
    ablation = outputs["first"]["ablation.json"]
    for key in (
        "correlations", "high_correlation_pairs", "leave_one_feature_out",
        "vocabulary_family", "provisional_headline_exclusions", "double_count_warnings",
    ):
        assert key in ablation
    assert "mtld" in ablation["vocabulary_family"]
    assert "hapax_ratio" in ablation["provisional_headline_exclusions"]
    backtest = outputs["first"]["backtest.json"]
    assert backtest["future_leakage_check"]["history_stable"] is True
    assert backtest["days"]
    assert {"headline_published", "source_coverage", "ci_low", "ci_high"} <= set(backtest["days"][0])
    sensitivity = outputs["first"]["sensitivity.json"]
    assert sensitivity["n_draws"] >= 20
    assert sensitivity["score_p05"] <= sensitivity["score_baseline"] <= sensitivity["score_p95"]
    drift = outputs["first"]["drift.json"]
    assert drift["movement"]
    assert {"score_effect", "composition_effect"} <= set(drift["movement"][0])

"""
AI Content Detector — applies the feature extractor to a corpus
and produces per-document aliveness scores and a corpus-level summary.

Works on pandas DataFrames from the silver layer.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .linguistic_features import extract_features, features_to_scores, resolve_feature_scaling


# ── Weights (must match config/config.yaml detection.weights) ────────────────

SCORABLE_FEATURES = {
    "ttr", "mtld", "entropy", "sentence_variance", "repetition",
    "zipf_deviation", "compression_ratio", "character_entropy",
    "hapax_ratio", "yules_k", "lexical_diversity",
}


def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    unknown = sorted(set(weights) - SCORABLE_FEATURES)
    if unknown:
        raise ValueError(f"Detector weights contain unknown features: {', '.join(unknown)}")
    if not weights or any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Detector weights must be finite non-negative values")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise ValueError("Detector weights must sum to 1.0")
    return dict(weights)


def load_scoring_config(config_path: str = "config/config.yaml"):
    with open(config_path) as handle:
        detection = yaml.safe_load(handle).get("detection", {})
    version = detection.get("scoring_version")
    if not version:
        raise ValueError("detection.scoring_version is required")
    scaling = resolve_feature_scaling(detection.get("feature_scaling") or {})
    return version, validate_weights(detection.get("weights", {})), scaling


def _score_document(text: str, version: str, weights: Dict[str, float], scaling: Dict) -> Dict:
    features = extract_features(text)
    sub_scores = features_to_scores(features, scaling)
    composite = sum(sub_scores.get(name, 0.5) * weight
                    for name, weight in weights.items()) * 100.0
    return {
        "aliveness_score": round(float(np.clip(composite, 0.0, 100.0)), 2),
        "scoring_version": version,
        **{f"feat_{key}": value for key, value in features.items()},
        **{f"score_{key}": round(value, 4) for key, value in sub_scores.items()},
    }


def score_document(text: str, timestamps: Optional[List[float]] = None,
                   weights: Optional[Dict[str, float]] = None,
                   config_path: str = "config/config.yaml") -> Dict:
    """
    Score a single document.
    Returns dict with raw features, sub-scores, and composite aliveness_score [0, 100].
    """
    version, configured_weights, scaling = load_scoring_config(config_path)
    resolved_weights = validate_weights(weights) if weights else configured_weights
    return _score_document(text, version, resolved_weights, scaling)


def score_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    weights: Optional[Dict[str, float]] = None,
    batch_size: int = 500,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Vectorised scoring of a silver-layer DataFrame.
    Adds aliveness_score + all feature/sub-score columns in place.
    """
    print(f"[DETECTOR] Scoring {len(df):,} documents …")
    version, configured_weights, scaling = load_scoring_config(config_path)
    resolved_weights = validate_weights(weights) if weights else configured_weights
    print(f"[DETECTOR] Scoring regime: {version}")

    results: List[Dict] = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        for _, row in batch.iterrows():
            text = str(row.get(text_col, "") or "")
            scored = _score_document(text, version, resolved_weights, scaling)
            results.append(scored)
        if (i // batch_size) % 10 == 0:
            print(f"  … {min(i + batch_size, len(df)):,}/{len(df):,}")

    score_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), score_df], axis=1)


def corpus_summary(scored_df: pd.DataFrame) -> Dict:
    """Aggregate corpus-level statistics from a scored DataFrame."""
    if scored_df.empty or "aliveness_score" not in scored_df.columns:
        return {}

    scores = scored_df["aliveness_score"].dropna()
    return {
        "n_documents": len(scored_df),
        "mean_aliveness": round(float(scores.mean()), 2),
        "median_aliveness": round(float(scores.median()), 2),
        "std_aliveness": round(float(scores.std()), 2),
        "pct_below_50": round(float((scores < 50).mean() * 100), 1),
        "pct_below_30": round(float((scores < 30).mean() * 100), 1),
        "per_source": (
            scored_df.groupby("source")["aliveness_score"]
            .agg(["mean", "median", "count"])
            .round(2)
            .rename(columns={"mean": "mean_score", "median": "median_score", "count": "n_docs"})
            .to_dict(orient="index")
        ) if "source" in scored_df.columns else {},
        "per_category": (
            scored_df.groupby("category")["aliveness_score"]
            .agg(["mean", "median", "count"])
            .round(2)
            .rename(columns={"mean": "mean_score", "median": "median_score", "count": "n_docs"})
            .to_dict(orient="index")
        ) if "category" in scored_df.columns else {},
    }

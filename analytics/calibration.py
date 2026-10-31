"""Labeled documents and a regularized logistic baseline.

Common schema, one document per row, JSONL or CSV:

Required columns: doc_id, text, label.
label is "human" or "generated". Numeric 1 means generated and 0 means human.

Optional columns used for stratification and out-of-distribution checks:
source, genre, language, domain, generator.

length_bin is derived from the token count: short below 80, medium below 200,
otherwise long. The model predicts the generated class. Its coefficients are an
interpretable baseline. They do not replace the headline weights when the file
has fewer than 200 documents.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from detection.linguistic_features import (
    PROVISIONAL_FEATURE_SCALING,
    extract_features,
    features_to_scores,
    tokenise,
)

REQUIRED_COLUMNS = ("doc_id", "text", "label")
OPTIONAL_COLUMNS = ("source", "genre", "language", "domain", "generator")
MODEL_FEATURES = [
    "character_entropy", "compression_ratio", "entropy", "hapax_ratio",
    "lexical_diversity", "mtld", "repetition", "sentence_variance",
    "ttr", "yules_k", "zipf_deviation",
]
VOCABULARY_FAMILY = ["hapax_ratio", "lexical_diversity", "mtld", "ttr", "yules_k"]
PROVISIONAL_EXCLUSIONS = ["hapax_ratio", "lexical_diversity", "ttr", "yules_k"]
HEADLINE_VOCABULARY_FEATURE = "mtld"
MIN_RELIABLE_DOCUMENTS = 200
CORRELATION_FLAG = 0.7
OOD_RULE = (
    "A document is out of distribution when its source, genre, language, or "
    "length bin was absent from the training file. An out-of-distribution score "
    "is less reliable than a score inside that support."
)


def _normalize_label(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"human", "0"}:
            return 0
        if key in {"generated", "1"}:
            return 1
        return None
    if value in (0, 1) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1) and float(value).is_integer():
        return int(value)
    return None


def _length_bin(text: str) -> str:
    count = len(tokenise(str(text)))
    if count < 80:
        return "short"
    if count < 200:
        return "medium"
    return "long"


def load_labeled_dataset(path) -> pd.DataFrame:
    file_path = str(path)
    if file_path.endswith(".csv"):
        frame = pd.read_csv(file_path)
    else:
        frame = pd.read_json(file_path, lines=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Labeled dataset requires columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Labeled dataset is empty")
    frame = frame.copy()
    frame["label_name"] = frame["label"].map(
        lambda value: "generated" if _normalize_label(value) == 1 else (
            "human" if _normalize_label(value) == 0 else None))
    frame["y"] = frame["label"].map(_normalize_label)
    if frame["y"].isna().any():
        raise ValueError("label must be human, generated, 0, or 1")
    frame["y"] = frame["y"].astype(int)
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = "unknown"
        else:
            frame[column] = frame[column].fillna("unknown").astype(str)
    frame["doc_id"] = frame["doc_id"].astype(str)
    frame["length_bin"] = frame["text"].map(_length_bin)
    return frame.sort_values(["doc_id"]).reset_index(drop=True)


def feature_tables(frame: pd.DataFrame):
    raw_rows = []
    score_rows = []
    for text in frame["text"]:
        features = extract_features(str(text))
        raw_rows.append(features)
        score_rows.append(features_to_scores(features))
    raw = pd.DataFrame(raw_rows, index=frame.index)
    scores = pd.DataFrame(score_rows, index=frame.index)
    return raw, scores


def _round(value, digits=6):
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _metrics(y, probability) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    predicted = (probability >= 0.5).astype(int)
    result = {
        "n": int(len(y)),
        "n_generated": int(y.sum()) if len(y) else 0,
        "accuracy": _round(float((predicted == y).mean())) if len(y) else None,
        "brier": _round(float(np.mean((probability - y) ** 2))) if len(y) else None,
        "roc_auc": None,
        "log_loss": None,
    }
    if len(set(y.tolist())) > 1:
        result["roc_auc"] = _round(float(roc_auc_score(y, probability)))
        result["log_loss"] = _round(float(log_loss(y, probability, labels=[0, 1])))
    return result


def fit_logistic(matrix: pd.DataFrame, y, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0, l1_ratio=0.0, solver="lbfgs", random_state=int(seed), max_iter=500,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(matrix[MODEL_FEATURES].to_numpy(dtype=float), np.asarray(y, dtype=int))
    return model


def _probabilities(model, matrix: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(matrix[MODEL_FEATURES].to_numpy(dtype=float))[:, 1]


def stratified_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    result = {}
    for column in ("source", "genre", "language", "length_bin"):
        rows = []
        for key, group in frame.groupby(column, sort=True):
            metrics = _metrics(group["y"], probability[group.index])
            metrics["value"] = str(key)
            rows.append(metrics)
        result[column] = rows
    return result


def leave_one_group_out(frame, scores, column: str, seed: int) -> list:
    results = []
    for key in sorted(frame[column].astype(str).unique()):
        test_mask = frame[column].astype(str).eq(key)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        if train["y"].nunique() < 2 or test.empty:
            results.append({
                "held_out": str(key), "status": "skipped", "n": int(len(test)),
                "reason": "training fold needs both classes",
            })
            continue
        model = fit_logistic(scores.loc[train.index], train["y"], seed)
        metrics = _metrics(test["y"], _probabilities(model, scores.loc[test.index]))
        metrics.update(held_out=str(key), status="scored")
        results.append(metrics)
    return results


def recommend_scaling(raw: pd.DataFrame, n_documents: int) -> dict:
    """Percentile map that can replace detection.feature_scaling in config.

    The block stays provisional below 200 documents. Copy it into config only
    after reading the limitations in the calibration JSON.
    """
    scaling = {
        "provisional": True,
        "sufficient_sample": bool(n_documents >= MIN_RELIABLE_DOCUMENTS),
        "method": "training_percentile_10_90",
        "n_documents": int(n_documents),
    }
    specs = {
        "compression_ratio": "linear",
        "character_entropy": "linear",
        "lexical_diversity": "linear",
        "hapax_ratio": "divisor",
        "yules_k": "invert",
    }
    for name, kind in specs.items():
        values = raw[name].astype(float).to_numpy()
        low = float(np.quantile(values, 0.10))
        high = float(np.quantile(values, 0.90))
        span = max(high - low, 1e-6)
        if kind == "linear":
            scaling[name] = {"low": _round(low), "span": _round(span)}
        elif kind == "divisor":
            scaling[name] = {"divisor": _round(max(high, 1e-6))}
        else:
            scaling[name] = {"divisor": _round(max(high, 1e-6)), "invert": True}
    return scaling


def training_support(frame: pd.DataFrame) -> dict:
    return {
        column: sorted(frame[column].astype(str).unique())
        for column in ("source", "genre", "language", "length_bin", "domain", "generator")
    }


def ood_flags(training: pd.DataFrame, rows: pd.DataFrame) -> list:
    support = {
        column: set(training[column].astype(str))
        for column in ("source", "genre", "language", "length_bin")
    }
    work = rows.copy()
    if "length_bin" not in work.columns:
        if "text" in work.columns:
            work["length_bin"] = work["text"].map(_length_bin)
        else:
            work["length_bin"] = "unknown"
    flags = []
    for _, row in work.iterrows():
        item = {}
        for column, values in support.items():
            observed = str(row[column]) if column in row and pd.notna(row[column]) else "unknown"
            item[f"{column}_ood"] = observed not in values
        item["any_ood"] = any(item.values())
        flags.append(item)
    return flags


def build_calibration_report(frame, *, seed: int, weights: dict, input_sha256: str) -> dict:
    import sklearn

    raw, scores = feature_tables(frame)
    model = fit_logistic(scores, frame["y"], seed)
    probability = _probabilities(model, scores)
    n_documents = int(len(frame))
    limitations = [
        "Metrics describe this input file only.",
        "The regularized logistic regression is an interpretable baseline, not a calibrated probability of AI authorship.",
        "Headline weights still omit hapax ratio, Yule's K, lexical diversity and type-token ratio so the vocabulary-richness family is not counted four times.",
    ]
    if n_documents < MIN_RELIABLE_DOCUMENTS:
        limitations.append(
            f"The file has {n_documents} documents, below {MIN_RELIABLE_DOCUMENTS}. "
            "Feature scaling and coefficients stay provisional."
        )
    coefficients = {
        name: _round(float(coef))
        for name, coef in zip(MODEL_FEATURES, model.coef_.ravel())
    }
    return {
        "script": "calibrate_detector",
        "schema_version": "iai-v2-analysis-1",
        "seed": int(seed),
        "input_sha256": input_sha256,
        "n_documents": n_documents,
        "model": {
            "type": "sklearn.linear_model.LogisticRegression",
            "sklearn_version": sklearn.__version__,
            "C": 1.0,
            "l1_ratio": 0.0,
            "solver": "lbfgs",
            "random_state": int(seed),
            "positive_class": "generated",
            "features": list(MODEL_FEATURES),
            "coefficients": coefficients,
            "intercept": _round(float(model.intercept_[0])),
        },
        "metrics": _metrics(frame["y"], probability),
        "stratified": stratified_metrics(frame, probability),
        "leave_one_generator_out": leave_one_group_out(frame, scores, "generator", seed),
        "leave_one_domain_out": leave_one_group_out(frame, scores, "domain", seed),
        "feature_scaling": recommend_scaling(raw, n_documents),
        "recommended_weights": {
            "status": "retained_provisional_config",
            "reason": "Coefficients are not applied as headline weights below 200 labeled documents.",
            "weights": {name: _round(float(value), 12) for name, value in sorted(weights.items())},
        },
        "training_support": training_support(frame),
        "ood_rule": OOD_RULE,
        "provisional_scaling_defaults": {
            key: value for key, value in PROVISIONAL_FEATURE_SCALING.items() if key != "provisional"
        },
        "limitations": limitations,
    }


def correlation_matrix(scores: pd.DataFrame) -> dict:
    columns = [name for name in MODEL_FEATURES if name in scores.columns]
    corr = scores[columns].corr(method="pearson")
    payload = {}
    for left in columns:
        payload[left] = {}
        for right in columns:
            value = corr.loc[left, right]
            payload[left][right] = None if pd.isna(value) else _round(float(value))
    return payload


def high_correlation_pairs(scores: pd.DataFrame, threshold: float = CORRELATION_FLAG) -> list:
    columns = [name for name in MODEL_FEATURES if name in scores.columns]
    corr = scores[columns].corr(method="pearson")
    pairs = []
    for index, left in enumerate(columns):
        for right in columns[index + 1:]:
            value = corr.loc[left, right]
            if pd.isna(value) or abs(float(value)) < threshold:
                continue
            family = None
            if left in VOCABULARY_FAMILY and right in VOCABULARY_FAMILY:
                family = "vocabulary_richness"
            pairs.append({
                "family": family,
                "left": left,
                "r": _round(float(value)),
                "right": right,
            })
    return pairs


def double_count_warnings(pairs: list) -> list:
    names = set()
    for pair in pairs:
        if pair.get("family") != "vocabulary_richness":
            continue
        for name in (pair["left"], pair["right"]):
            if name != HEADLINE_VOCABULARY_FEATURE:
                names.add(name)
    return sorted(names)


def leave_one_feature_out(frame, scores, seed: int) -> list:
    full = fit_logistic(scores, frame["y"], seed)
    full_auc = _metrics(frame["y"], _probabilities(full, scores))["roc_auc"]
    rows = []
    for feature in MODEL_FEATURES:
        kept = [name for name in MODEL_FEATURES if name != feature]
        model = LogisticRegression(
            C=1.0, l1_ratio=0.0, solver="lbfgs", random_state=int(seed), max_iter=500,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(scores[kept].to_numpy(dtype=float), frame["y"].to_numpy(dtype=int))
        probability = model.predict_proba(scores[kept].to_numpy(dtype=float))[:, 1]
        auc = _metrics(frame["y"], probability)["roc_auc"]
        delta = None if auc is None or full_auc is None else _round(auc - full_auc)
        rows.append({"feature": feature, "roc_auc": auc, "roc_auc_delta": delta})
    return rows


def build_ablation_report(frame, *, seed: int, input_sha256: str) -> dict:
    _raw, scores = feature_tables(frame)
    pairs = high_correlation_pairs(scores)
    warnings_found = double_count_warnings(pairs)
    limitations = [
        "Correlations are measured on this input file only.",
        "The headline keeps MTLD as the single vocabulary-richness weight. Hapax ratio, Yule's K, lexical diversity and type-token ratio stay in the feature table for calibration.",
    ]
    if not warnings_found:
        limitations.append(
            "This file did not show a vocabulary-family correlation at or above 0.7. "
            "The headline exclusion is still the provisional policy."
        )
    return {
        "script": "feature_ablation",
        "schema_version": "iai-v2-analysis-1",
        "seed": int(seed),
        "input_sha256": input_sha256,
        "correlations": correlation_matrix(scores),
        "high_correlation_pairs": pairs,
        "leave_one_feature_out": leave_one_feature_out(frame, scores, seed),
        "vocabulary_family": list(VOCABULARY_FAMILY),
        "provisional_headline_exclusions": list(PROVISIONAL_EXCLUSIONS),
        "double_count_warnings": warnings_found,
        "correlation_threshold": CORRELATION_FLAG,
        "limitations": limitations,
    }


def weight_sensitivity(scores: pd.DataFrame, weights: dict, *, seed: int, n_draws: int = 100) -> dict:
    keys = sorted(weights)
    base = np.array([float(weights[key]) for key in keys], dtype=float)
    matrix = scores[keys].to_numpy(dtype=float)
    baseline_scores = matrix @ base * 100.0
    baseline = float(baseline_scores.mean())
    rng = np.random.default_rng(int(seed))
    draws = rng.dirichlet(np.maximum(base, 1e-6) * 80.0, size=int(n_draws))
    means = (matrix @ draws.T).mean(axis=0) * 100.0
    return {
        "baseline_weights": {key: _round(float(weights[key]), 12) for key in keys},
        "n_draws": int(n_draws),
        "score_baseline": _round(baseline),
        "score_p05": _round(float(np.quantile(means, 0.05))),
        "score_p50": _round(float(np.quantile(means, 0.50))),
        "score_p95": _round(float(np.quantile(means, 0.95))),
        "max_abs_delta": _round(float(np.max(np.abs(means - baseline)))),
    }

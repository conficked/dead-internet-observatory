"""
Distributional estimate of the AI-generated share of a corpus.

Each document is reduced to the set of vocabulary words it contains within a fixed
token window. Human and AI reference corpora give per-word occurrence probabilities,
and a target corpus is modelled as the mixture

    P(x) = (1 - alpha) * P_human(x) + alpha * P_ai(x)

alpha is fitted by maximum likelihood, following Liang et al. (2024), "Monitoring
AI-Modified Content at Scale". It estimates the share of documents, not a label per
document: per-document likelihood ratios from a bag-of-words model are overconfident,
the corpus-level maximum is not. Intervals are document bootstraps.
"""

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar

from detection.linguistic_features import tokenise

WINDOW_TOKENS = 150
MIN_LINE_WORDS = 8
_SPACE_RE = re.compile(r"\s+")


def main_text(text: str, min_line_words: int = MIN_LINE_WORDS) -> str:
    """Keep prose lines. Navigation, cookie banners and link lists are short lines,
    and only human crawl pages carry them, so they would separate the corpora on
    page furniture instead of authorship."""
    lines = (line.strip() for line in str(text or "").splitlines())
    return " ".join(line for line in lines if len(line.split()) >= min_line_words)


def document_window(text: str, window: int = WINDOW_TOKENS) -> Optional[List[str]]:
    """First `window` tokens of the prose, or None when the document is shorter.
    A fixed window keeps word occurrence from tracking document length."""
    tokens = tokenise(main_text(text))
    return tokens[:window] if len(tokens) >= window else None


def windows(texts: Iterable[str], window: int = WINDOW_TOKENS) -> List[List[str]]:
    return [w for w in (document_window(t, window) for t in texts) if w is not None]


class OccurrenceModel:
    """Per-word occurrence probabilities for the human and AI reference corpora."""

    def __init__(self, vocabulary: Sequence[str], p_human: np.ndarray, p_ai: np.ndarray):
        self.vocabulary = list(vocabulary)
        self.index = {word: i for i, word in enumerate(self.vocabulary)}
        self.p_human = np.asarray(p_human, dtype=float)
        self.p_ai = np.asarray(p_ai, dtype=float)

    @classmethod
    def fit(cls, human: Sequence[Sequence[str]], ai: Sequence[Sequence[str]],
            min_doc_frequency: float = 0.01, smoothing: float = 1.0,
            allowed: Optional[set] = None) -> "OccurrenceModel":
        if not human or not ai:
            raise ValueError("Both reference corpora need documents")
        human_df = _document_frequencies(human)
        ai_df = _document_frequencies(ai)
        vocabulary = sorted(
            word for word in set(human_df) | set(ai_df)
            if (allowed is None or word in allowed)
            and max(human_df.get(word, 0) / len(human), ai_df.get(word, 0) / len(ai))
            >= min_doc_frequency)
        if not vocabulary:
            raise ValueError("No word reaches min_doc_frequency in either corpus")

        def probabilities(frequencies, n):
            counts = np.array([frequencies.get(word, 0) for word in vocabulary], dtype=float)
            return (counts + smoothing) / (n + 2 * smoothing)

        return cls(vocabulary, probabilities(human_df, len(human)), probabilities(ai_df, len(ai)))

    @classmethod
    def from_matrices(cls, vocabulary, human_matrix, ai_matrix, human_weights=None,
                      ai_weights=None, smoothing: float = 1.0) -> "OccurrenceModel":
        """Refit probabilities on a fixed vocabulary; weights are bootstrap counts."""
        def probabilities(matrix, weights):
            weights = np.ones(matrix.shape[0]) if weights is None else np.asarray(weights, float)
            counts = np.asarray(matrix.T @ weights).ravel()
            return (counts + smoothing) / (weights.sum() + 2 * smoothing)

        return cls(vocabulary, probabilities(human_matrix, human_weights),
                   probabilities(ai_matrix, ai_weights))

    def matrix(self, documents: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        """Binary document-by-vocabulary occurrence matrix."""
        rows, cols = [], []
        for i, document in enumerate(documents):
            present = {self.index[w] for w in document if w in self.index}
            rows.extend([i] * len(present))
            cols.extend(present)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                 shape=(len(documents), len(self.vocabulary)))

    def log_ratio_matrix(self, matrix) -> np.ndarray:
        """log P(x | AI) - log P(x | human) for each row of an occurrence matrix."""
        present = (np.log(self.p_ai) - np.log1p(-self.p_ai)) - (
            np.log(self.p_human) - np.log1p(-self.p_human))
        base = np.log1p(-self.p_ai).sum() - np.log1p(-self.p_human).sum()
        return np.asarray(matrix @ present).ravel() + base

    def log_likelihoods(self, documents: Sequence[Sequence[str]]):
        """Per-document log P(x | human) and log P(x | AI)."""
        human_present = np.log(self.p_human) - np.log1p(-self.p_human)
        ai_present = np.log(self.p_ai) - np.log1p(-self.p_ai)
        human_base = np.log1p(-self.p_human).sum()
        ai_base = np.log1p(-self.p_ai).sum()
        lh = np.empty(len(documents))
        la = np.empty(len(documents))
        for i, document in enumerate(documents):
            present = np.fromiter(
                {self.index[w] for w in document if w in self.index}, dtype=int)
            lh[i] = human_base + human_present[present].sum()
            la[i] = ai_base + ai_present[present].sum()
        return lh, la

    def top_words(self, n: int = 20) -> Dict[str, List[str]]:
        log_ratio = np.log(self.p_ai) - np.log(self.p_human)
        order = np.argsort(log_ratio)
        return {"ai": [self.vocabulary[i] for i in order[::-1][:n]],
                "human": [self.vocabulary[i] for i in order[:n]]}


def _document_frequencies(documents):
    frequencies: Dict[str, int] = {}
    for document in documents:
        for word in set(document):
            frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies


def mixture_log_likelihood(alpha: float, log_ratio: np.ndarray) -> float:
    """Sum of log((1 - alpha) + alpha * exp(log P_ai - log P_human)), constant dropped."""
    alpha = float(np.clip(alpha, 1e-12, 1 - 1e-12))
    return float(np.logaddexp(np.log1p(-alpha), np.log(alpha) + log_ratio).sum())


def mle_alpha(log_ratio: np.ndarray) -> float:
    log_ratio = np.asarray(log_ratio, dtype=float)
    if log_ratio.size == 0:
        raise ValueError("No documents to estimate from")
    # The likelihood is concave in alpha; the boundaries are checked by their slope.
    ratio = np.exp(np.clip(log_ratio, -700, 700))
    if np.sum(ratio - 1) <= 0:
        return 0.0
    if np.sum(1 - 1 / ratio) >= 0:
        return 1.0
    result = minimize_scalar(lambda a: -mixture_log_likelihood(a, log_ratio),
                             bounds=(0.0, 1.0), method="bounded",
                             options={"xatol": 1e-6})
    return float(result.x)


def estimate_alpha(log_ratio: np.ndarray, resamples: int = 200, seed: int = 0,
                   level: float = 0.95) -> Dict:
    log_ratio = np.asarray(log_ratio, dtype=float)
    rng = np.random.default_rng(seed)
    point = mle_alpha(log_ratio)
    draws = np.array([mle_alpha(rng.choice(log_ratio, size=log_ratio.size))
                      for _ in range(resamples)])
    tail = (1 - level) / 2
    return {"alpha": point, "ci_low": float(np.quantile(draws, tail)),
            "ci_high": float(np.quantile(draws, 1 - tail)), "n": int(log_ratio.size)}


def _mixed_estimates(rng, human_ratio, ai_ratio, alpha, sample_size, repeats):
    n_ai = int(round(alpha * sample_size))
    return np.array([
        mle_alpha(np.concatenate([
            rng.choice(human_ratio, size=sample_size - n_ai),
            rng.choice(ai_ratio, size=n_ai) if n_ai else np.empty(0)]))
        for _ in range(repeats)])


def injection_test(human_ratio: np.ndarray, ai_ratio: np.ndarray,
                   alphas: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5),
                   sample_size: int = 2000, repeats: int = 50, seed: int = 0,
                   correct: Optional[Callable[[float], float]] = None) -> List[Dict]:
    """Mix held-out human and AI documents at known shares and re-estimate them.
    `correct` maps a raw estimate to a calibrated one before scoring."""
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in alphas:
        estimates = _mixed_estimates(rng, human_ratio, ai_ratio, alpha, sample_size, repeats)
        if correct is not None:
            estimates = np.array([correct(value) for value in estimates])
        rows.append({"true_alpha": alpha, "mean_estimate": float(estimates.mean()),
                     "bias": float(estimates.mean() - alpha),
                     "rmse": float(np.sqrt(np.mean((estimates - alpha) ** 2)))})
    return rows


class ConfidenceBelt:
    """Calibration of the raw estimator from simulated mixtures at known shares.

    For each true share on a grid, mixtures of `sample_size` documents give the
    distribution of the raw estimate. The corrected point inverts the mean curve; the
    corrected interval is every true share whose central band contains the observed
    raw value (a Neyman construction), so it carries the estimator's bias and noise.
    It corrects only the bias these calibration pools can show: drift between the
    calibration era and a target is not visible to it.
    """

    FINE_GRID = np.r_[np.arange(0, 0.2, 0.0025), np.arange(0.2, 0.5001, 0.01)]

    def __init__(self, grid, mean, low, high):
        self.grid = np.asarray(grid, dtype=float)
        self.mean = np.maximum.accumulate(np.asarray(mean, dtype=float))
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)

    @classmethod
    def build(cls, human_ratio, ai_ratio, sample_size, grid=None, repeats=200, seed=0,
              level=0.95):
        grid = cls.FINE_GRID if grid is None else np.asarray(grid, dtype=float)
        rng = np.random.default_rng(seed)
        tail = (1 - level) / 2
        mean, low, high = [], [], []
        for alpha in grid:
            estimates = _mixed_estimates(rng, human_ratio, ai_ratio, alpha, sample_size, repeats)
            mean.append(estimates.mean())
            low.append(np.quantile(estimates, tail))
            high.append(np.quantile(estimates, 1 - tail))
        return cls(grid, mean, low, high)

    def point(self, raw: float) -> float:
        if raw <= self.mean[0]:
            return 0.0
        if raw >= self.mean[-1]:
            return float(self.grid[-1])
        return float(np.interp(raw, self.mean, self.grid))

    def interval(self, raw: float):
        covered = self.grid[(self.low <= raw) & (raw <= self.high)]
        if covered.size == 0:
            nearest = self.grid[np.argmin(np.minimum(abs(self.low - raw), abs(self.high - raw)))]
            return float(nearest), float(nearest)
        return float(covered.min()), float(covered.max())

    def correct(self, estimate: Dict) -> Dict:
        low, high = self.interval(estimate["alpha"])
        return dict(estimate, raw_alpha=estimate["alpha"], raw_ci_low=estimate["ci_low"],
                    raw_ci_high=estimate["ci_high"], alpha=self.point(estimate["alpha"]),
                    ci_low=low, ci_high=high)


def dictionary_vocabulary(path: str) -> set:
    """Lowercase dictionary entries. Capitalised entries are proper nouns, so site and
    brand names drop out along with concatenated page furniture such as 'twittershare'."""
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return {line.strip() for line in handle
                if line.strip().isalpha() and line.strip().islower()}


def placebo_verdict(estimates: Sequence[Dict], tolerance: float = 0.02) -> Dict:
    """Pre-ChatGPT text must estimate near zero. Fails when any period's interval
    lies wholly above `tolerance`, or the pooled mean exceeds it."""
    alphas = [row["alpha"] for row in estimates]
    flagged = [row for row in estimates if row["ci_low"] > tolerance]
    mean = float(np.mean(alphas)) if alphas else float("nan")
    return {"passed": bool(alphas) and not flagged and mean <= tolerance,
            "tolerance": tolerance, "mean_alpha": mean, "max_alpha": max(alphas, default=None),
            "periods_above_tolerance": [row.get("period") for row in flagged]}


def trend_placebo(dates: Sequence, values: Sequence[float], resamples: int = 2000,
                  seed: int = 0) -> Dict:
    """Per-year OLS slope of a pre-ChatGPT series with a period bootstrap. A score that
    tracks AI content has no business trending before AI text existed."""
    years = (np.asarray(dates, dtype="datetime64[D]").astype(float) / 365.25)
    values = np.asarray(values, dtype=float)
    slope = float(np.polyfit(years, values, 1)[0])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        pick = rng.integers(0, len(years), len(years))
        if np.ptp(years[pick]) > 0:
            draws.append(np.polyfit(years[pick], values[pick], 1)[0])
    low, high = (float(q) for q in np.quantile(draws, [0.025, 0.975]))
    return {"slope_per_year": slope, "ci_low": low, "ci_high": high,
            "passed": low <= 0 <= high, "periods": int(len(years))}


def cluster_resample(values: np.ndarray, clusters: np.ndarray, rng) -> np.ndarray:
    """Two-stage bootstrap: resample clusters, then documents within each drawn cluster.
    With one cluster it is an ordinary document bootstrap."""
    labels = np.unique(clusters)
    picks = []
    for label in rng.choice(labels, size=labels.size):
        members = np.flatnonzero(clusters == label)
        picks.append(rng.choice(members, size=members.size))
    return values[np.concatenate(picks)]


def replicate_intervals(model: OccurrenceModel, fit_human, fit_ai, calibration_human,
                        calibration_ai, periods: Dict[str, tuple], sample_size: int,
                        replicates: int = 40, belt_repeats: int = 60, seed: int = 0,
                        level: float = 0.95) -> Dict[str, Dict]:
    """Intervals that carry every resampled source of error.

    Each replicate refits the occurrence model on bootstrapped reference corpora,
    rebuilds the confidence belt from bootstrapped calibration pools, and re-estimates
    every period from a two-stage (cluster, document) resample. `periods` maps a name
    to (occurrence matrix, cluster labels). Vocabulary selection is held fixed.
    """
    rng = np.random.default_rng(seed)
    raw = {name: [] for name in periods}
    corrected = {name: [] for name in periods}
    for replicate in range(replicates):
        weights = [np.bincount(rng.integers(0, m.shape[0], m.shape[0]), minlength=m.shape[0])
                   for m in (fit_human, fit_ai)]
        refit = OccurrenceModel.from_matrices(model.vocabulary, fit_human, fit_ai, *weights)
        cal_h = refit.log_ratio_matrix(calibration_human)
        cal_a = refit.log_ratio_matrix(calibration_ai)
        belt = ConfidenceBelt.build(rng.choice(cal_h, cal_h.size), rng.choice(cal_a, cal_a.size),
                                    sample_size, repeats=belt_repeats,
                                    seed=int(rng.integers(2 ** 31)))
        for name, (matrix, clusters) in periods.items():
            value = mle_alpha(cluster_resample(refit.log_ratio_matrix(matrix),
                                               np.asarray(clusters), rng))
            raw[name].append(value)
            corrected[name].append(belt.point(value))
    tail = (1 - level) / 2
    return {name: {"raw_ci": [float(np.quantile(raw[name], q)) for q in (tail, 1 - tail)],
                   "ci": [float(np.quantile(corrected[name], q)) for q in (tail, 1 - tail)],
                   "replicates": len(corrected[name])}
            for name in periods}

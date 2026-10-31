"""
Detection Scientist — linguistic feature extractor.

Computes a battery of statistical signals that distinguish
human-authored text from AI-generated or bot-produced content.

No large LLMs required. All methods are O(n) to O(n log n)
and run fast on CPU with standard Python/NumPy.

Signals implemented:
  ttr              — Type-Token Ratio (vocabulary richness)
  mtld             — Measure of Textual Lexical Diversity (McCarthy 2005)
  entropy          — Shannon entropy of word unigrams
  sentence_variance — CV of sentence lengths (AI text is suspiciously uniform)
  repetition       — Ratio of repeated bigrams to total bigrams
  zipf_deviation   — Deviation from Zipf's power-law for natural language
  burstiness       — Goh-Barabasi burstiness of inter-event times.
                       Document scoring does not use it. Source-level summaries do.

Compression ratio, hapax ratio and Yule's K are averaged over fixed windows so a
repeated passage keeps the same value as it grows. The full-document values are
retained as *_raw features. The [0, 1] maps for the five newer features are
provisional until a labeled calibration file replaces them.
"""

import math
import re
import zlib
from typing import Dict, List, Optional

import numpy as np

# Fixed windows make repeated text produce the same feature value at different lengths.
TOKEN_WINDOW = 50
COMPRESSION_WINDOW_BYTES = 256

# Uncalibrated maps from raw features to [0, 1]. Calibration JSON can replace them.
PROVISIONAL_FEATURE_SCALING = {
    "provisional": True,
    "compression_ratio": {"low": 0.25, "span": 0.55},
    "character_entropy": {"low": 3.0, "span": 2.0},
    "hapax_ratio": {"divisor": 0.7},
    "yules_k": {"divisor": 300.0, "invert": True},
    "lexical_diversity": {"low": 0.2, "span": 0.5},
}


# ── Text tokenisation ─────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b[a-z]{2,}\b")
_SENT_RE = re.compile(r"[^.!?]+[.!?]+", re.MULTILINE)


def tokenise(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


# ── Individual feature functions ──────────────────────────────────────────────

def compute_ttr(tokens: List[str]) -> float:
    """
    Type-Token Ratio: unique_words / total_words.
    Range [0, 1]. Higher = more diverse vocabulary.
    Naive TTR is length-sensitive; we cap at 200 tokens for comparability.
    """
    if not tokens:
        return 0.0
    sample = tokens[:200]
    return len(set(sample)) / len(sample)


def compute_mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    Scans forward until TTR drops below threshold, counts each as a 'factor'.
    MTLD = len(tokens) / factor_count.
    Higher = more lexically diverse. Typical human prose: 70-120.
    """
    if len(tokens) < 10:
        return 0.0

    def _mtld_forward(toks: List[str]) -> float:
        factors = 0.0
        types: set = set()
        token_count = 0
        start = 0
        for i, tok in enumerate(toks):
            types.add(tok)
            token_count += 1
            current_ttr = len(types) / token_count
            if current_ttr <= threshold:
                factors += 1.0
                types = set()
                token_count = 0
                start = i + 1
        # Partial factor
        if token_count > 0:
            partial_ttr = len(types) / token_count
            if partial_ttr < 1.0:
                completion = (1.0 - partial_ttr) / (1.0 - threshold)
                factors += completion
        return len(toks) / factors if factors > 0 else len(toks)

    # Average forward + backward for stability
    fwd = _mtld_forward(tokens)
    bwd = _mtld_forward(tokens[::-1])
    return (fwd + bwd) / 2.0


def compute_entropy(tokens: List[str]) -> float:
    """
    Shannon entropy of the unigram word distribution.
    H = -Σ p(w) log2 p(w).
    Higher = more uniform / information-dense text.
    AI text tends to be slightly lower entropy (repetitive phrases).
    """
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    n = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _compression_ratio(raw: bytes) -> float:
    if not raw:
        return 0.0
    return len(zlib.compress(raw)) / len(raw)


def compute_compression_ratio(text: str, window: int = COMPRESSION_WINDOW_BYTES) -> float:
    """Mean zlib ratio over non-overlapping byte windows.

    A short document uses its whole byte string. A longer document drops a trailing
    partial window so repeated text stays comparable across lengths.
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    if len(raw) <= window:
        return _compression_ratio(raw)
    ratios = [
        _compression_ratio(raw[start:start + window])
        for start in range(0, len(raw) - window + 1, window)
    ]
    return float(np.mean(ratios))


def compute_character_entropy(text: str) -> float:
    if not text:
        return 0.0
    frequencies: Dict[str, int] = {}
    for character in text:
        frequencies[character] = frequencies.get(character, 0) + 1
    length = len(text)
    return -sum((count / length) * math.log2(count / length)
                for count in frequencies.values())


def _hapax_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    frequencies: Dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    return sum(count == 1 for count in frequencies.values()) / len(frequencies)


def _window_mean(tokens: List[str], window: int, measure) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return float(measure(tokens))
    return float(np.mean([
        measure(tokens[start:start + window])
        for start in range(0, len(tokens) - window + 1, window)
    ]))


def compute_hapax_ratio(tokens: List[str], window: int = TOKEN_WINDOW) -> float:
    """Mean hapax-type ratio over non-overlapping token windows."""
    return _window_mean(tokens, window, _hapax_ratio)


def _yules_k(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    frequencies: Dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    length = len(tokens)
    frequency_moment = sum(count * count for count in frequencies.values())
    return 10_000.0 * (frequency_moment - length) / (length * length)


def compute_yules_k(tokens: List[str], window: int = TOKEN_WINDOW) -> float:
    """Mean Yule's K over non-overlapping token windows."""
    return _window_mean(tokens, window, _yules_k)


def compute_lexical_diversity(tokens: List[str], window: int = TOKEN_WINDOW) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([
        len(set(tokens[start:start + window])) / window
        for start in range(len(tokens) - window + 1)
    ]))


def compute_sentence_variance(text: str) -> float:
    """
    Coefficient of Variation (CV) of sentence lengths in words.
    Human writing has high CV (mix of short and long sentences).
    AI text is suspiciously uniform: CV << 0.5.
    Returns CV directly; we'll invert/normalise in the scorer.
    """
    sents = sentences(text)
    if len(sents) < 3:
        return 0.5   # neutral default for very short texts
    lengths = [len(tokenise(s)) for s in sents]
    arr = np.array(lengths, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std() / mean)   # CV


def compute_repetition(tokens: List[str]) -> float:
    """
    Ratio of repeated bigrams to total bigrams.
    AI text frequently repeats stock phrases → higher repetition.
    Returns [0, 1]; 0 = no repetition, 1 = all bigrams repeated.
    """
    if len(tokens) < 4:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_freq: Dict[tuple, int] = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1
    repeated = sum(v - 1 for v in bigram_freq.values() if v > 1)
    return repeated / len(bigrams)


def compute_zipf_deviation(tokens: List[str]) -> float:
    """
    Measures how well the word frequency distribution fits Zipf's law.
    Natural language obeys: freq(rank r) ∝ 1/r^α  (α ≈ 1).
    We fit log-log slope; deviation from -1 indicates unnatural text.
    Returns 'alignment score' [0, 1]; 1 = perfect Zipf.
    """
    if len(tokens) < 30:
        return 0.5

    freq: Dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1

    sorted_freqs = sorted(freq.values(), reverse=True)[:50]  # top 50
    if len(sorted_freqs) < 5:
        return 0.5

    ranks = np.arange(1, len(sorted_freqs) + 1, dtype=float)
    freqs_arr = np.array(sorted_freqs, dtype=float)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs_arr)

    # Linear regression of log-log
    coeffs = np.polyfit(log_ranks, log_freqs, 1)
    slope = coeffs[0]

    # Ideal slope ≈ -1.0; we score based on distance from ideal
    deviation = abs(slope - (-1.0))
    # Map to [0, 1]: 0 deviation → 1.0 score; 2.0 deviation → 0.0
    score = max(0.0, 1.0 - deviation / 2.0)
    return float(score)


def compute_burstiness(timestamps_seconds: List[float]) -> float:
    """
    Goh-Barabasi (2008) burstiness parameter B from inter-event times.
    B = (σ - μ) / (σ + μ) ∈ [-1, 1].
    B ≈ +1 → very bursty (human bursts of activity).
    B ≈ -1 → perfectly regular (bot-like scheduling).
    B ≈  0 → Poisson / random.
    Returns a normalised aliveness score [0, 1].
    """
    if len(timestamps_seconds) < 3:
        return 0.5   # neutral

    ts = sorted(timestamps_seconds)
    iets = np.diff(ts)
    iets = iets[iets > 0]

    if len(iets) < 2:
        return 0.5

    mu = iets.mean()
    sigma = iets.std()

    if mu + sigma == 0:
        return 0.5

    B = (sigma - mu) / (sigma + mu)
    # Map from [-1, 1] to [0, 1]: bursty (positive) = more alive
    return float((B + 1.0) / 2.0)


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_features(
    text: str,
    timestamps: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute all linguistic features for one text document.
    Returns a dict of raw feature values (not yet normalised to scores).
    """
    tokens = tokenise(text)

    features = {
        "ttr": compute_ttr(tokens),
        "mtld": compute_mtld(tokens),
        "entropy": compute_entropy(tokens),
        "sentence_variance": compute_sentence_variance(text),
        "repetition": compute_repetition(tokens),
        "zipf_deviation": compute_zipf_deviation(tokens),
        "compression_ratio": compute_compression_ratio(text),
        "compression_ratio_raw": _compression_ratio(text.encode("utf-8")),
        "character_entropy": compute_character_entropy(text),
        "hapax_ratio": compute_hapax_ratio(tokens),
        "hapax_ratio_raw": _hapax_ratio(tokens),
        "yules_k": compute_yules_k(tokens),
        "yules_k_raw": _yules_k(tokens),
        "lexical_diversity": compute_lexical_diversity(tokens),
        "token_count": len(tokens),
        "sentence_count": len(sentences(text)),
        "avg_word_length": (
            sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
        ),
    }

    return features


def resolve_feature_scaling(overrides: Optional[Dict] = None) -> Dict:
    """Merge a calibration block onto the provisional scaling map."""
    resolved = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in PROVISIONAL_FEATURE_SCALING.items()
    }
    for key, value in (overrides or {}).items():
        if key == "provisional":
            resolved["provisional"] = bool(value)
        elif isinstance(value, dict) and isinstance(resolved.get(key), dict):
            merged = dict(resolved[key])
            merged.update(value)
            resolved[key] = merged
    return resolved


def _linear_unit(value: float, spec: Dict) -> float:
    span = float(spec.get("span", 1.0))
    if span == 0:
        return 0.0
    return min(1.0, max(0.0, (float(value) - float(spec.get("low", 0.0))) / span))


def _divisor_unit(value: float, spec: Dict) -> float:
    divisor = float(spec.get("divisor", 1.0))
    if divisor == 0:
        return 0.0
    scaled = min(1.0, max(0.0, float(value) / divisor))
    if spec.get("invert"):
        return max(0.0, 1.0 - scaled)
    return scaled


def features_to_scores(features: Dict[str, float], scaling: Optional[Dict] = None) -> Dict[str, float]:
    """
    Normalise raw features to [0, 1] aliveness sub-scores.
    Higher sub-score always means "more alive / human-like".

    Scaling for compression, character entropy, hapax ratio, Yule's K and
    lexical diversity comes from PROVISIONAL_FEATURE_SCALING unless a calibration
    block replaces those maps.
    """
    scaling = resolve_feature_scaling(scaling)
    scores: Dict[str, float] = {}

    # TTR: 0.3–0.7 is typical human; cap and scale
    ttr = features.get("ttr", 0.0)
    scores["ttr"] = min(1.0, max(0.0, (ttr - 0.2) / 0.5))

    # MTLD: 40 = low diversity, 120 = high. Cap at 120.
    mtld = features.get("mtld", 0.0)
    scores["mtld"] = min(1.0, max(0.0, (mtld - 20.0) / 100.0))

    # Entropy: typical English ≈ 8–11 bits. Scale.
    ent = features.get("entropy", 0.0)
    scores["entropy"] = min(1.0, max(0.0, (ent - 5.0) / 8.0))

    # Sentence variance: human CV ≈ 0.5–1.5. Low CV → low score.
    sv = features.get("sentence_variance", 0.0)
    scores["sentence_variance"] = min(1.0, max(0.0, sv / 1.2))

    # Repetition: lower is better (more human). Invert.
    rep = features.get("repetition", 0.0)
    scores["repetition"] = max(0.0, 1.0 - (rep / 0.3))

    # Zipf: already [0, 1]
    scores["zipf_deviation"] = features.get("zipf_deviation", 0.5)

    scores["compression_ratio"] = _linear_unit(
        features.get("compression_ratio", 0.0), scaling["compression_ratio"])
    scores["character_entropy"] = _linear_unit(
        features.get("character_entropy", 0.0), scaling["character_entropy"])
    scores["hapax_ratio"] = _divisor_unit(
        features.get("hapax_ratio", 0.0), scaling["hapax_ratio"])
    scores["yules_k"] = _divisor_unit(
        features.get("yules_k", 0.0), scaling["yules_k"])
    scores["lexical_diversity"] = _linear_unit(
        features.get("lexical_diversity", 0.0), scaling["lexical_diversity"])

    return scores

"""Deterministic cluster bootstrap intervals.

Clusters are resampled with replacement. A source with fewer than two clusters
does not receive an interval, because a single cluster cannot show between-cluster
spread. Seeds come from numpy's Generator and are stable across processes.
"""

import hashlib

import numpy as np


def stable_seed(base: int, key: str) -> int:
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    extra = int.from_bytes(digest[:4], "little")
    return (int(base) + extra) % (2 ** 32)


def cluster_bootstrap_mean(sums, counts, *, seed: int, resamples: int = 400):
    """Return (low, high) for the mean, resampling clusters with replacement."""
    sums = np.asarray(sums, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if len(sums) < 2 or len(sums) != len(counts):
        return (None, None)
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, len(sums), size=(int(resamples), len(sums)))
    means = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return (float(low), float(high))


def composite_cluster_bootstrap(source_clusters, weights, *, seed: int, resamples: int = 400):
    """Propagate within-source cluster resamples through fixed source weights.

    Sources are not themselves resampled: the panel weights are a design, not a
    sample. Every included source needs at least two clusters.
    """
    names = [
        name for name in sorted(source_clusters)
        if name in weights and weights[name] > 0 and len(source_clusters[name][0]) >= 2
    ]
    if len(names) != len(source_clusters) or not names:
        return (None, None)
    rng = np.random.default_rng(int(seed))
    columns = []
    used = []
    for name in names:
        sums = np.asarray(source_clusters[name][0], dtype=float)
        counts = np.asarray(source_clusters[name][1], dtype=float)
        index = rng.integers(0, len(sums), size=(int(resamples), len(sums)))
        columns.append(sums[index].sum(axis=1) / counts[index].sum(axis=1))
        used.append(float(weights[name]))
    weight = np.asarray(used, dtype=float)
    weight = weight / weight.sum()
    composite = weight @ np.vstack(columns)
    low, high = np.quantile(composite, [0.025, 0.975])
    return (float(low), float(high))

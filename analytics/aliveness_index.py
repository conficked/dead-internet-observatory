"""
Index & Analytics Agent — Internet Aliveness Index (IAI) computation.

The IAI is a composite 0–100 score measuring how much of the sampled
internet appears authentically human versus synthetic/automated.

Scores describe heuristic human-like signals, not calibrated AI probabilities.
Daily source/category summaries are stored alongside document scores so delta
ingestion can recompute exact statistics without counting a document twice.
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .bootstrap import cluster_bootstrap_mean, composite_cluster_bootstrap, stable_seed


# ── Database schema ───────────────────────────────────────────────────────────

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS ingested_docs (
    doc_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    aliveness_score REAL NOT NULL,
    domain TEXT
);
CREATE INDEX IF NOT EXISTS ingested_docs_group ON ingested_docs(date, source, category);
CREATE TABLE IF NOT EXISTS daily_index (
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    n_docs INTEGER,
    mean_score REAL,
    median_score REAL,
    std_score REAL,
    pct_below_50 REAL,
    bot_fraction REAL,
    aliveness_index REAL,
    PRIMARY KEY (date, source, category)
);

CREATE TABLE IF NOT EXISTS composite_index (
    date TEXT PRIMARY KEY,
    aliveness_index REAL,
    smoothed_index REAL,
    n_docs INTEGER,
    anomaly_flag INTEGER DEFAULT 0,
    anomaly_reason TEXT
);

CREATE TABLE IF NOT EXISTS domain_scores (
    date TEXT NOT NULL,
    domain TEXT NOT NULL,
    category TEXT,
    mean_score REAL,
    n_docs INTEGER,
    PRIMARY KEY (date, domain)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS scoring_regimes (
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    scoring_version TEXT NOT NULL,
    PRIMARY KEY (entity_type, entity_key)
);
CREATE TABLE IF NOT EXISTS index_quality (
    date TEXT PRIMARY KEY,
    source_coverage REAL NOT NULL,
    coverage_sufficient INTEGER NOT NULL,
    sample_sufficient INTEGER NOT NULL,
    eligible_sources INTEGER NOT NULL,
    required_sources INTEGER NOT NULL,
    ci_low REAL,
    ci_high REAL,
    n_clusters INTEGER,
    current_regime_docs INTEGER,
    noncurrent_docs INTEGER,
    regime_sufficient INTEGER
);
CREATE TABLE IF NOT EXISTS source_interval (
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    mean_score REAL,
    ci_low REAL,
    ci_high REAL,
    n_docs INTEGER,
    n_clusters INTEGER,
    scoring_version TEXT,
    PRIMARY KEY (date, source)
);
"""

QUALITY_COLUMNS = {
    "ci_low": "REAL",
    "ci_high": "REAL",
    "n_clusters": "INTEGER",
    "current_regime_docs": "INTEGER",
    "noncurrent_docs": "INTEGER",
    "regime_sufficient": "INTEGER",
}


class AlivenessIndexEngine:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)

        self.db_path = self.config["storage"].get("db_path", "./data/observatory.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        cfg = self.config["analytics"]
        self.smooth_window: int = cfg.get("index_smoothing_window", 7)
        self.anomaly_z: float = cfg.get("anomaly_zscore_threshold", 2.5)
        self.source_weights = cfg.get("source_weights", {})
        self.core_sources = cfg.get("core_source_panel", list(self.source_weights))
        self.minimum_source_samples = int(cfg.get("minimum_source_samples", 1))
        self.minimum_total_samples = int(cfg.get("minimum_total_samples", 10))
        self.minimum_coverage = float(cfg.get("minimum_coverage", 1.0))
        self.bootstrap_resamples = max(20, min(int(cfg.get("bootstrap_resamples", 400)), 2000))
        self.bootstrap_seed = int(cfg.get("bootstrap_seed", 0))
        self.scoring_version = self.config.get("detection", {}).get(
            "scoring_version", "legacy-v1")
        if self.source_weights:
            missing_weights = set(self.core_sources) - set(self.source_weights)
            if missing_weights:
                raise ValueError(f"Core sources missing weights: {sorted(missing_weights)}")
            if not np.isclose(sum(self.source_weights.values()), 1.0, atol=1e-9):
                raise ValueError("analytics.source_weights must sum to 1.0")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("analytics.minimum_coverage must be in (0, 1]")

        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(CREATE_TABLES)
            existing = {row[1] for row in conn.execute("PRAGMA table_info(index_quality)")}
            for name, declaration in QUALITY_COLUMNS.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE index_quality ADD COLUMN {name} {declaration}")
            conn.execute(
                "INSERT OR IGNORE INTO meta VALUES ('created_at', ?)",
                (datetime.now(timezone.utc).isoformat(),)
            )

    # ── Index computation ─────────────────────────────────────────────────────

    @staticmethod
    def _valid_dated_docs(df):
        df = df.copy()
        raw_dates = df.get("created_dt", pd.Series(pd.NaT, index=df.index))
        dates = pd.to_datetime(raw_dates, errors="coerce", utc=True, format="mixed")
        missing = raw_dates.isna()
        invalid = dates.isna() & ~missing
        future = dates > pd.Timestamp(datetime.now(timezone.utc))
        logging.getLogger(__name__).info(
            "[INDEX] Dropped %d null created_dt rows, %d invalid created_dt rows and %d future created_dt rows",
            int(missing.sum()), int(invalid.sum()), int(future.sum()))
        df = df.loc[~(missing | invalid | future)].copy()
        df["date"] = dates.loc[df.index].dt.strftime("%Y-%m-%d")
        df["aliveness_score"] = pd.to_numeric(df["aliveness_score"], errors="coerce")
        return df.loc[np.isfinite(df["aliveness_score"])].copy()

    def ingest_scored_df(self, df: pd.DataFrame, replace_affected_dates: bool = False):
        """Replace supplied source/category groups (or entire affected dates)."""
        self._ingest(df, delta=False, replace_dates=replace_affected_dates)

    def ingest_scored_delta(self, df: pd.DataFrame):
        """Insert unseen IDs; merge legacy summaries approximately, others exactly."""
        self._ingest(df, delta=True)

    def _ingest(self, df, *, delta, replace_dates=False):
        if df.empty or "aliveness_score" not in df.columns:
            return
        df = self._valid_dated_docs(df)
        if df.empty:
            return
        if "doc_id" not in df or df["doc_id"].isna().any() or df["doc_id"].eq("").any():
            raise ValueError("Scored documents require nonempty doc_id for idempotent ingestion")
        df = df.drop_duplicates("doc_id")
        groups = list(df.groupby(["date", "source", "category"]).groups)
        with self._conn() as conn:
            affected = set()
            domains = set()
            legacy = {}
            inserted_scores = {}
            if delta:
                for key in groups:
                    old = conn.execute(
                        "SELECT * FROM daily_index WHERE date=? AND source=? AND category=?", key
                    ).fetchone()
                    ledger_count = conn.execute(
                        "SELECT COUNT(*) FROM ingested_docs WHERE date=? AND source=? AND category=?", key
                    ).fetchone()[0]
                    # A count gap also preserves legacy history on subsequent deltas.
                    if old is not None and (ledger_count == 0 or old["n_docs"] > ledger_count):
                        legacy[key] = old
            if not delta:
                if replace_dates:
                    for date in df["date"].unique():
                        for table in ("ingested_docs", "daily_index", "domain_scores", "composite_index"):
                            conn.execute(f"DELETE FROM {table} WHERE date = ?", (date,))
                else:
                    for key in groups:
                        domains.update((key[0], row[0]) for row in conn.execute(
                            "SELECT DISTINCT domain FROM ingested_docs WHERE date=? AND source=? AND category=? AND domain IS NOT NULL", key))
                        conn.execute("DELETE FROM ingested_docs WHERE date=? AND source=? AND category=?", key)
                        conn.execute("DELETE FROM daily_index WHERE date=? AND source=? AND category=?", key)
            for row in df.to_dict("records"):
                key = (row["date"], row["source"], row["category"])
                domain = row.get("domain")
                if pd.isna(domain):
                    domain = None
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO ingested_docs VALUES (?,?,?,?,?,?)",
                    (str(row["doc_id"]), *key, float(row["aliveness_score"]), domain))
                if cursor.rowcount:
                    version = str(row.get("scoring_version") or self.scoring_version)
                    conn.execute(
                        "INSERT OR REPLACE INTO scoring_regimes VALUES ('document', ?, ?)",
                        (str(row["doc_id"]), version))
                    affected.add(key)
                    inserted_scores.setdefault(key, []).append(float(row["aliveness_score"]))
                    if domain:
                        domains.add((row["date"], domain))
            for key in affected:
                if key in legacy:
                    old = legacy[key]
                    scores = pd.Series(inserted_scores[key])
                    n_old, n_delta = old["n_docs"], len(scores)
                    n_docs = n_old + n_delta
                    delta_mean = float(scores.mean())
                    mean_raw = (n_old * old["mean_score"] + n_delta * delta_mean) / n_docs
                    median = (n_old * old["median_score"] + n_delta * float(scores.median())) / n_docs
                    pct = (n_old * old["pct_below_50"] + n_delta * float((scores < 50).mean() * 100)) / n_docs
                    # Pool sample variances, including the between-group mean shift.
                    m2 = (max(n_old - 1, 0) * (old["std_score"] or 0.0) ** 2
                          + float(((scores - delta_mean) ** 2).sum())
                          + (old["mean_score"] - delta_mean) ** 2 * n_old * n_delta / n_docs)
                    std = (m2 / (n_docs - 1)) ** 0.5 if n_docs > 1 else None
                else:
                    scores = pd.Series([r[0] for r in conn.execute(
                        "SELECT aliveness_score FROM ingested_docs WHERE date=? AND source=? AND category=?", key)])
                    n_docs = len(scores)
                    mean_raw = float(scores.mean())
                    median = float(scores.median())
                    std = float(scores.std()) if n_docs > 1 else None
                    pct = float((scores < 50).mean() * 100)
                mean = round(mean_raw, 3)
                conn.execute("INSERT OR REPLACE INTO daily_index VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (*key, n_docs, mean, round(median, 3),
                              round(std, 3) if std is not None else None,
                              round(pct, 2), 0.0, mean))
                versions = sorted({
                    str(value) for value in df.loc[
                        df["date"].eq(key[0]) & df["source"].eq(key[1]) &
                        df["category"].eq(key[2]), "scoring_version"
                    ].dropna().unique()
                }) if "scoring_version" in df else []
                version = versions[0] if len(versions) == 1 else (
                    f"mixed[{','.join(versions)}]" if versions else self.scoring_version)
                conn.execute(
                    "INSERT OR REPLACE INTO scoring_regimes VALUES ('daily', ?, ?)",
                    ("\x1f".join(key), version))
            for date, domain in domains:
                rows = conn.execute("SELECT aliveness_score, category FROM ingested_docs WHERE date=? AND domain=?",
                                    (date, domain)).fetchall()
                if not rows:
                    conn.execute("DELETE FROM domain_scores WHERE date=? AND domain=?", (date, domain))
                    continue
                conn.execute("INSERT OR REPLACE INTO domain_scores VALUES (?,?,?,?,?)",
                             (date, domain, rows[0][1], round(float(np.mean([r[0] for r in rows])), 3), len(rows)))
            conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('total_indexed_count', ?)",
                (str(conn.execute("SELECT COUNT(*) FROM ingested_docs").fetchone()[0]),))
        legacy_count = len(affected.intersection(legacy))
        if legacy_count:
            logging.getLogger(__name__).warning(
                "[INDEX] Merged %d legacy daily groups approximately", legacy_count)
        self._recompute_composite()

    def _source_cluster_stats(self, clusters, date, source, categories):
        """Return (sums, counts) per domain cluster, or None when the ledger is short."""
        if clusters.empty or not categories:
            return None
        subset = clusters[
            clusters["date"].astype(str).eq(str(date))
            & clusters["source"].astype(str).eq(str(source))
            & clusters["category"].astype(str).isin([str(name) for name in categories])
            & clusters["scoring_version"].astype(str).eq(self.scoring_version)
        ]
        if subset.empty:
            return None
        for category, expected in categories.items():
            observed = int(subset.loc[subset["category"].astype(str).eq(str(category)), "n"].sum())
            if observed != int(expected):
                return None
        grouped = subset.groupby("cluster_id")[["score_sum", "n"]].sum()
        return grouped["score_sum"].to_numpy(dtype=float), grouped["n"].to_numpy(dtype=float)

    def _recompute_composite(self):
        """Recompute the headline from current-regime source means.

        A daily group enters the headline only when its scoring version is exactly
        the configured regime. Mixed and unversioned groups stay in daily_index
        for diagnosis and do not add coverage. Cluster bootstrap intervals use
        domain clusters from the document ledger. A short ledger or a single
        cluster leaves the interval empty.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT d.date, d.source, d.category, d.mean_score, d.n_docs,
                          COALESCE(r.scoring_version, 'unversioned') AS scoring_version
                   FROM daily_index d
                   LEFT JOIN scoring_regimes r
                     ON r.entity_type = 'daily'
                    AND r.entity_key = d.date || char(31) || d.source || char(31) || d.category"""
            ).fetchall()
            cluster_rows = conn.execute(
                """SELECT d.date AS date, d.source AS source, d.category AS category,
                          CASE WHEN d.domain IS NULL OR d.domain = ''
                               THEN 'category:' || d.category
                               ELSE 'domain:' || d.domain END AS cluster_id,
                          COALESCE(r.scoring_version, 'unversioned') AS scoring_version,
                          SUM(d.aliveness_score) AS score_sum,
                          COUNT(*) AS n
                   FROM ingested_docs d
                   LEFT JOIN scoring_regimes r
                     ON r.entity_type = 'document' AND r.entity_key = d.doc_id
                   GROUP BY d.date, d.source, d.category, cluster_id, scoring_version"""
            ).fetchall()

        if not rows:
            with self._conn() as conn:
                conn.execute("DELETE FROM composite_index")
                conn.execute("DELETE FROM index_quality")
                conn.execute("DELETE FROM source_interval")
                conn.execute("DELETE FROM scoring_regimes WHERE entity_type = 'composite'")
            return

        daily = pd.DataFrame([dict(row) for row in rows])
        clusters = pd.DataFrame([dict(row) for row in cluster_rows]) if cluster_rows else pd.DataFrame(
            columns=["date", "source", "category", "cluster_id", "scoring_version", "score_sum", "n"])
        current = self.scoring_version
        source_rows = []
        noncurrent_by_date = {}
        categories_by_source = {}
        for (date, source), group in daily.groupby(["date", "source"], sort=False):
            other = group[group["scoring_version"] != current]
            noncurrent_by_date[date] = noncurrent_by_date.get(date, 0) + int(other["n_docs"].sum())
            current_groups = group[group["scoring_version"] == current]
            if current_groups.empty:
                continue
            categories_by_source[(date, source)] = {
                str(row.category): int(row.n_docs) for row in current_groups.itertuples()
            }
            source_rows.append({
                "date": date,
                "source": source,
                "mean_score": float(np.average(
                    current_groups["mean_score"], weights=current_groups["n_docs"])),
                "n_docs": int(current_groups["n_docs"].sum()),
                "scoring_version": current,
            })
        sources = pd.DataFrame(source_rows, columns=[
            "date", "source", "mean_score", "n_docs", "scoring_version"])

        records = []
        interval_rows = []
        for date in sorted(daily["date"].unique()):
            observed = sorted(daily.loc[daily["date"].eq(date), "source"].astype(str).unique())
            panel = list(self.core_sources) if self.core_sources else observed
            weights = self.source_weights or {
                source: 1 / len(panel) for source in panel} if panel else {}
            group = sources[sources["date"].eq(date) & sources["source"].isin(panel)].copy()
            if not group.empty:
                group["weight"] = group["source"].map(weights).astype(float)
                eligible = group[group["n_docs"] >= self.minimum_source_samples].copy()
            else:
                eligible = group
            panel_weight = sum(float(weights.get(source, 0.0)) for source in panel)
            eligible_weight = float(eligible["weight"].sum()) if not eligible.empty else 0.0
            coverage = eligible_weight / panel_weight if panel_weight else 0.0
            n_docs = int(group["n_docs"].sum()) if not group.empty else 0
            coverage_sufficient = coverage >= self.minimum_coverage
            sample_sufficient = n_docs >= self.minimum_total_samples
            score = None
            if coverage_sufficient and sample_sufficient and eligible_weight and not eligible.empty:
                score = float(np.average(eligible["mean_score"], weights=eligible["weight"]))
            noncurrent = int(noncurrent_by_date.get(date, 0))
            source_clusters = {}
            n_clusters = 0
            interval_ready = True
            for row in eligible.itertuples():
                stats = self._source_cluster_stats(
                    clusters, date, row.source, categories_by_source.get((date, row.source), {}))
                low = high = None
                cluster_count = 0
                if stats is None:
                    interval_ready = False
                else:
                    sums, counts = stats
                    cluster_count = int(len(sums))
                    n_clusters += cluster_count
                    low, high = cluster_bootstrap_mean(
                        sums, counts,
                        seed=stable_seed(self.bootstrap_seed, f"{date}:{row.source}"),
                        resamples=self.bootstrap_resamples,
                    )
                    if low is None:
                        interval_ready = False
                    else:
                        source_clusters[row.source] = stats
                interval_rows.append({
                    "date": date, "source": row.source, "mean_score": float(row.mean_score),
                    "ci_low": low, "ci_high": high, "n_docs": int(row.n_docs),
                    "n_clusters": cluster_count, "scoring_version": current,
                })
            ci_low = ci_high = None
            if score is not None and interval_ready and len(source_clusters) == len(eligible):
                ci_low, ci_high = composite_cluster_bootstrap(
                    source_clusters, weights,
                    seed=stable_seed(self.bootstrap_seed, f"{date}:composite"),
                    resamples=self.bootstrap_resamples,
                )
            records.append({
                "date": date,
                "aliveness_index": score,
                "n_docs": n_docs,
                "source_coverage": coverage,
                "coverage_sufficient": int(coverage_sufficient),
                "sample_sufficient": int(sample_sufficient),
                "eligible_sources": int(len(eligible)),
                "required_sources": int(len(panel)),
                "scoring_version": current if n_docs else "withheld:no-current-regime",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_clusters": n_clusters,
                "current_regime_docs": n_docs,
                "noncurrent_docs": noncurrent,
                "regime_sufficient": int(noncurrent == 0),
            })

        composite = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        smoothed = composite["aliveness_index"].rolling(
            window=self.smooth_window, min_periods=1).mean()
        composite["smoothed_index"] = smoothed.where(
            composite["aliveness_index"].notna()).round(3)

        rolling_mean = composite["aliveness_index"].rolling(30, min_periods=7).mean()
        rolling_std = composite["aliveness_index"].rolling(30, min_periods=7).std()
        z_scores = (composite["aliveness_index"] - rolling_mean) / (rolling_std + 1e-9)
        composite["anomaly_flag"] = (z_scores.abs() > self.anomaly_z).astype(int)
        composite["anomaly_reason"] = np.where(
            z_scores > self.anomaly_z, "spike",
            np.where(z_scores < -self.anomaly_z, "drop", "")
        )

        def _optional(value, digits=None):
            if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
                return None
            number = float(value)
            return round(number, digits) if digits is not None else number

        with self._conn() as conn:
            conn.execute("DELETE FROM composite_index")
            conn.execute("DELETE FROM index_quality")
            conn.execute("DELETE FROM source_interval")
            conn.execute("DELETE FROM scoring_regimes WHERE entity_type = 'composite'")
            for _, row in composite.iterrows():
                date = str(row["date"])
                conn.execute(
                    """INSERT INTO composite_index VALUES (?,?,?,?,?,?)
                       ON CONFLICT(date) DO UPDATE SET
                           aliveness_index = excluded.aliveness_index,
                           smoothed_index  = excluded.smoothed_index,
                           n_docs          = excluded.n_docs,
                           anomaly_flag    = excluded.anomaly_flag,
                           anomaly_reason  = excluded.anomaly_reason""",
                    (
                        date,
                        _optional(row["aliveness_index"], 3),
                        _optional(row["smoothed_index"], 3),
                        int(row["n_docs"]) if not pd.isna(row["n_docs"]) else 0,
                        int(row["anomaly_flag"]),
                        row["anomaly_reason"],
                    )
                )
                conn.execute(
                    """INSERT INTO index_quality
                       (date, source_coverage, coverage_sufficient, sample_sufficient,
                        eligible_sources, required_sources, ci_low, ci_high, n_clusters,
                        current_regime_docs, noncurrent_docs, regime_sufficient)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        date, round(float(row["source_coverage"]), 6),
                        int(row["coverage_sufficient"]), int(row["sample_sufficient"]),
                        int(row["eligible_sources"]), int(row["required_sources"]),
                        _optional(row["ci_low"]), _optional(row["ci_high"]),
                        int(row["n_clusters"]), int(row["current_regime_docs"]),
                        int(row["noncurrent_docs"]), int(row["regime_sufficient"]),
                    )
                )
                conn.execute(
                    "INSERT INTO scoring_regimes VALUES ('composite', ?, ?)",
                    (date, row["scoring_version"]))
            for row in interval_rows:
                conn.execute(
                    """INSERT INTO source_interval
                       (date, source, mean_score, ci_low, ci_high, n_docs, n_clusters, scoring_version)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        row["date"], row["source"], round(float(row["mean_score"]), 3),
                        _optional(row["ci_low"]), _optional(row["ci_high"]),
                        int(row["n_docs"]), int(row["n_clusters"]), row["scoring_version"],
                    )
                )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_composite_timeline(self, days: int = 365) -> pd.DataFrame:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, q.source_coverage, q.coverage_sufficient,
                          q.sample_sufficient, q.eligible_sources, q.required_sources,
                          q.ci_low, q.ci_high, q.n_clusters, q.current_regime_docs,
                          q.noncurrent_docs, q.regime_sufficient,
                          r.scoring_version
                   FROM composite_index c
                   LEFT JOIN index_quality q ON q.date = c.date
                   LEFT JOIN scoring_regimes r
                     ON r.entity_type = 'composite' AND r.entity_key = c.date
                   WHERE c.date >= ? ORDER BY c.date""",
                (str(cutoff),)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_source_intervals(self, date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM source_interval"
        params: tuple = ()
        if date:
            query += " WHERE date = ?"
            params = (date,)
        query += " ORDER BY date, source"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def get_source_breakdown(self, date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM daily_index"
        params: tuple = ()
        if date:
            query += " WHERE date = ?"
            params = (date,)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_top_domains(self, n: int = 50, date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM domain_scores"
        params: tuple = ()
        if date:
            query += " WHERE date = ?"
            params = (date,)
        query += " ORDER BY n_docs DESC LIMIT ?"
        params = params + (n,)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_current_score(self) -> Optional[float]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT smoothed_index FROM composite_index
                   WHERE date <= ? AND smoothed_index IS NOT NULL
                   ORDER BY date DESC LIMIT 1""",
                (datetime.now(timezone.utc).date().isoformat(),)
            ).fetchone()
        return float(row["smoothed_index"]) if row else None

    def reconcile_future_dates(self):
        """Report existing future-dated rows without changing stored observations."""
        today = datetime.now(timezone.utc).date().isoformat()
        with self._conn() as conn:
            return {table: conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE date > ?", (today,)
            ).fetchone()[0] for table in (
                "ingested_docs", "daily_index", "composite_index", "domain_scores")}

    def get_total_docs(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(n_docs), 0) FROM daily_index"
            ).fetchone()
        return int(row[0])

    def get_meta(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str):
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

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
"""


class AlivenessIndexEngine:

    DEFAULT_SOURCE_WEIGHTS = {
        "common_crawl":  0.30,
        "reddit":        0.15,
        "news":          0.10,
        "wikipedia":     0.10,
        "wayback":       0.08,
        "hackernews":    0.06,
        "bluesky":       0.05,
        "youtube":       0.05,
        "fourchan":      0.04,
        "steam":         0.03,
        "mastodon":      0.03,
        "stackoverflow": 0.03,
        "linkedin":      0.02,
        "github":        0.02,
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)

        self.db_path = self.config["storage"].get("db_path", "./data/observatory.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        cfg = self.config["analytics"]
        self.smooth_window: int = cfg.get("index_smoothing_window", 7)
        self.anomaly_z: float = cfg.get("anomaly_zscore_threshold", 2.5)

        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(CREATE_TABLES)
            conn.execute(
                "INSERT OR IGNORE INTO meta VALUES ('created_at', ?)",
                (datetime.now(timezone.utc).isoformat(),)
            )

    # ── Index computation ─────────────────────────────────────────────────────

    @staticmethod
    def _valid_dated_docs(df):
        df = df.copy()
        dates = pd.to_datetime(df.get("created_dt", pd.Series(pd.NaT, index=df.index)),
                               errors="coerce", utc=True, format="mixed")
        missing = dates.isna()
        future = dates > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        logging.getLogger(__name__).info(
            "[INDEX] Dropped %d null created_dt rows and %d future created_dt rows",
            int(missing.sum()), int(future.sum()))
        df = df.loc[~(missing | future)].copy()
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
            for date, domain in domains:
                rows = conn.execute("SELECT aliveness_score, category FROM ingested_docs WHERE date=? AND domain=?",
                                    (date, domain)).fetchall()
                if not rows:
                    conn.execute("DELETE FROM domain_scores WHERE date=? AND domain=?", (date, domain))
                    continue
                conn.execute("INSERT OR REPLACE INTO domain_scores VALUES (?,?,?,?,?)",
                             (date, domain, rows[0][1], round(float(np.mean([r[0] for r in rows])), 3), len(rows)))
        legacy_count = len(affected.intersection(legacy))
        if legacy_count:
            logging.getLogger(__name__).warning(
                "[INDEX] Merged %d legacy daily groups approximately", legacy_count)
        self._recompute_composite()

    def _recompute_composite(self):
        """Recompute the composite daily index from daily_index table."""
        with self._conn() as conn:
            rows = conn.execute("SELECT date, source, mean_score, n_docs FROM daily_index").fetchall()

        if not rows:
            with self._conn() as conn:
                conn.execute("DELETE FROM composite_index")
            return

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])

        def weighted_mean(group: pd.DataFrame) -> float:
            total_weight = 0.0
            weighted_sum = 0.0
            for _, row in group.iterrows():
                w = self.DEFAULT_SOURCE_WEIGHTS.get(row["source"], 0.1)
                weighted_sum += row["mean_score"] * w
                total_weight += w
            return weighted_sum / total_weight if total_weight > 0 else 50.0

        n_docs_series = df.groupby("date")["n_docs"].sum()

        composite = (
            df.groupby("date")
            .apply(weighted_mean)
            .reset_index()
            .rename(columns={0: "aliveness_index"})
            .sort_values("date")
        )

        composite["n_docs"] = composite["date"].map(n_docs_series).fillna(0).apply(lambda x: int(float(x)))

        # Drop days with too few docs — single-doc outliers distort the composite
        composite = composite[composite["n_docs"] >= 10].reset_index(drop=True)

        # Rolling smoothing
        composite["smoothed_index"] = (
            composite["aliveness_index"]
            .rolling(window=self.smooth_window, min_periods=1, center=True)
            .mean()
            .round(3)
        )

        # Anomaly detection via rolling z-score
        rolling_mean = composite["aliveness_index"].rolling(30, min_periods=7).mean()
        rolling_std  = composite["aliveness_index"].rolling(30, min_periods=7).std()
        z_scores = (composite["aliveness_index"] - rolling_mean) / (rolling_std + 1e-9)
        composite["anomaly_flag"] = (z_scores.abs() > self.anomaly_z).astype(int)
        composite["anomaly_reason"] = np.where(
            z_scores > self.anomaly_z, "spike",
            np.where(z_scores < -self.anomaly_z, "drop", "")
        )

        with self._conn() as conn:
            conn.execute("DELETE FROM composite_index")
            for _, row in composite.iterrows():
                conn.execute(
                    """INSERT INTO composite_index VALUES (?,?,?,?,?,?)
                       ON CONFLICT(date) DO UPDATE SET
                           aliveness_index = excluded.aliveness_index,
                           smoothed_index  = excluded.smoothed_index,
                           n_docs          = excluded.n_docs,
                           anomaly_flag    = excluded.anomaly_flag,
                           anomaly_reason  = excluded.anomaly_reason""",
                    (
                        str(row["date"].date()),
                        round(float(row["aliveness_index"]), 3),
                        round(float(row["smoothed_index"]), 3),
                        int(row["n_docs"]) if not pd.isna(row["n_docs"]) else 0,
                        int(row["anomaly_flag"]),
                        row["anomaly_reason"],
                    )
                )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_composite_timeline(self, days: int = 365) -> pd.DataFrame:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM composite_index WHERE date >= ? ORDER BY date",
                (str(cutoff),)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

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
                "SELECT smoothed_index FROM composite_index ORDER BY date DESC LIMIT 1"
            ).fetchone()
        return float(row["smoothed_index"]) if row else None

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

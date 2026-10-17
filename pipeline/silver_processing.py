"""
Silver → Gold processing pipeline.

Incremental: only scores documents not already present in the gold layer.
New docs are appended to scored.parquet; the durable registry and SQLite index
are updated with each run.
"""

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from analytics.aliveness_index import AlivenessIndexEngine

MAX_NEW_DOCS = 6_000
MAX_NEW_CC_DOCS = 50_000


def cc_gold_counts(gold_path):
    if not gold_path.exists():
        return {}
    frame = pd.read_parquet(gold_path, columns=['source', 'crawl_partition'])
    counts = frame.loc[frame.source.eq('common_crawl'), 'crawl_partition'].dropna().str.replace('/', '-', regex=False).value_counts()
    return {cid: int(n) for cid, n in counts.items()}


def persist_cc_status(gold_path, conn):
    from datetime import datetime, timezone
    counts = cc_gold_counts(gold_path)
    conn.execute('CREATE TABLE IF NOT EXISTS cc_crawl_status (crawl_id TEXT PRIMARY KEY, gold_rows INTEGER, updated_at TEXT)')
    conn.execute('DELETE FROM cc_crawl_status')
    conn.executemany('INSERT INTO cc_crawl_status VALUES (?, ?, ?)',
                     [(cid, n, datetime.now(timezone.utc).isoformat()) for cid, n in counts.items()])


def select_new_docs(frame, counts, *, max_cc=MAX_NEW_CC_DOCS, per_crawl=10000, max_other=MAX_NEW_DOCS):
    cc_mask = frame['source'].eq('common_crawl')
    other = frame.loc[~cc_mask]
    if 'created_dt' in other:
        other = other.sort_values('created_dt', ascending=False)
    groups = []
    for cid, group in frame.loc[cc_mask].groupby('crawl_partition', sort=True):
        room = max(0, per_crawl - counts.get(str(cid).replace('/', '-'), 0))
        groups.append(group.head(room).copy().assign(_round=range(min(len(group), room))))
    cc = pd.concat(groups).sort_values('_round', kind='stable').head(max_cc).drop(columns='_round') if groups else frame.iloc[:0]
    return pd.concat([cc, other.head(max_other)], ignore_index=True)


def append_scored(gold_path, scored):
    """Merge the delta without materialising the historical corpus in pandas."""
    delta = pa.Table.from_pandas(
        scored.drop_duplicates(subset=["doc_id"], keep="last"), preserve_index=False
    )
    if not gold_path.exists():
        pq.write_table(delta, gold_path)
        return len(delta)

    temporary_path = gold_path.with_suffix(".tmp.parquet")
    try:
        with pq.ParquetFile(gold_path) as previous:
            schema = pa.unify_schemas(
                [previous.schema_arrow, delta.schema], promote_options="permissive"
            ).remove_metadata()

            def align(table):
                return pa.Table.from_arrays(
                    [table[field.name].cast(field.type) if field.name in table.column_names
                     else pa.nulls(len(table), type=field.type) for field in schema],
                    schema=schema,
                )

            delta_ids = pa.array(scored["doc_id"].dropna().unique())
            count = 0
            with pq.ParquetWriter(temporary_path, schema) as writer:
                for batch in previous.iter_batches(batch_size=10_000):
                    table = pa.Table.from_batches([batch])
                    table = table.filter(pc.invert(
                        pc.is_in(table["doc_id"], value_set=delta_ids)
                    ))
                    writer.write_table(align(table))
                    count += len(table)
                writer.write_table(align(delta))
                count += len(delta)
        os.replace(temporary_path, gold_path)
        return count
    finally:
        temporary_path.unlink(missing_ok=True)


class SilverToGoldPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)
        data_root = Path(self.config["storage"]["local_path"])
        self.silver_root = data_root / self.config["storage"]["silver_path"]
        self.gold_root   = data_root / self.config["storage"]["gold_path"]
        self.gold_root.mkdir(parents=True, exist_ok=True)
        self.engine = AlivenessIndexEngine(config_path)

    def run(self, source_file: str = "combined.parquet"):
        from detection.ai_content_detector import score_dataframe, corpus_summary

        silver_path = self.silver_root / source_file
        if not silver_path.exists():
            print(f"[GOLD] Silver file not found: {silver_path}")
            return

        print(f"[GOLD] Loading {silver_path} …")
        silver_df = pd.read_parquet(silver_path)
        print(f"[GOLD] {len(silver_df):,} docs in silver")

        # ── Load already-scored doc_ids ───────────────────────────────────────
        gold_path = self.gold_root / "scored.parquet"
        registry_path = self.gold_root / "doc_registry.parquet"
        local_existing_ids = set()
        if gold_path.exists():
            local_existing_ids = set(pd.read_parquet(gold_path, columns=["doc_id"])["doc_id"].dropna())

        registry_ids = set()
        if registry_path.exists():
            registry_ids = set(
                pd.read_parquet(registry_path, columns=["doc_id"])["doc_id"].dropna()
            )
        else:
            raise RuntimeError(
                "Missing gold/doc_registry.parquet; restore persistent state before scoring"
            )
        registry_ids.update(local_existing_ids)
        if registry_ids:
            pd.DataFrame({"doc_id": sorted(registry_ids)}).to_parquet(
                registry_path, index=False, engine="pyarrow"
            )
            with self.engine._conn() as conn:
                conn.execute(
                    """INSERT INTO meta (key, value) VALUES ('total_scored_count', ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                    (str(len(registry_ids)),),
                )
            print(f"[GOLD] Persisted {len(registry_ids):,} registry IDs")

        existing_ids = registry_ids

        if existing_ids:
            print(
                f"[GOLD] {len(existing_ids):,} docs already scored "
                f"({len(local_existing_ids):,} scored, {len(registry_ids):,} registry)"
            )
        else:
            print("[GOLD] No prior scored docs found — scoring full silver batch")

        new_df = silver_df[~silver_df["doc_id"].isin(existing_ids)].copy()
        del silver_df, local_existing_ids, existing_ids
        new_df = new_df.drop_duplicates(subset=["doc_id"], keep="last")
        print(f"[GOLD] {len(new_df):,} new docs to score")

        if new_df.empty:
            print("[GOLD] Nothing new — index already up to date")
            with self.engine._conn() as conn:
                persist_cc_status(gold_path, conn)
            return

        cc_config = self.config.get('sources', {}).get('common_crawl', {})
        new_df = select_new_docs(new_df, cc_gold_counts(gold_path),
                                 max_cc=cc_config.get('max_new_docs_per_run', MAX_NEW_CC_DOCS),
                                 per_crawl=cc_config.get('max_rows_per_crawl', 10000))

        # Reload the durable IDs after scoring instead of retaining millions of
        # Python strings alongside the model and historical gold batches.
        del registry_ids

        # ── Score, with perplexity disabled for common_crawl (too slow) ───────
        orig_perplexity = os.environ.get("ENABLE_PERPLEXITY", "")
        if "source" in new_df.columns:
            cc_mask = new_df["source"] == "common_crawl"
        else:
            cc_mask = pd.Series(False, index=new_df.index)

        cc_only = new_df[cc_mask].copy()
        non_cc  = new_df[~cc_mask].copy()
        del new_df, cc_mask

        scored_parts = []
        if not cc_only.empty:
            print(f"[GOLD] Scoring {len(cc_only):,} CC docs (perplexity disabled)")
            os.environ["ENABLE_PERPLEXITY"] = ""
            try:
                scored_parts.append(score_dataframe(cc_only))
            finally:
                os.environ["ENABLE_PERPLEXITY"] = orig_perplexity

        if not non_cc.empty:
            print(f"[GOLD] Scoring {len(non_cc):,} non-CC docs")
            scored_parts.append(score_dataframe(non_cc))

        scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()

        if scored.empty:
            print("[GOLD] No scored output — refreshing crawl status")
            with self.engine._conn() as conn:
                persist_cc_status(gold_path, conn)
            return

        summary = corpus_summary(scored)
        print(f"[GOLD] Summary: {summary}")

        # ── Append to gold parquet ────────────────────────────────────────────
        gold_count = append_scored(gold_path, scored)
        print(f"[GOLD] ✓ scored.parquet now has {gold_count:,} docs")

        registry_ids = set(
            pd.read_parquet(registry_path, columns=["doc_id"])["doc_id"].dropna()
        )
        registry_ids.update(scored["doc_id"].dropna().unique())
        pd.DataFrame({"doc_id": sorted(registry_ids)}).to_parquet(
            registry_path, index=False, engine="pyarrow"
        )
        print(f"[GOLD] ✓ doc_registry.parquet now has {len(registry_ids):,} docs")

        self.engine.ingest_scored_delta(scored)
        with self.engine._conn() as conn:
            conn.execute(
                """INSERT INTO meta (key, value) VALUES ('total_scored_count', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (str(len(registry_ids)),),
            )
        with self.engine._conn() as conn:
            persist_cc_status(gold_path, conn)
        print("[GOLD] ✓ SQLite index updated")


if __name__ == "__main__":
    SilverToGoldPipeline().run()

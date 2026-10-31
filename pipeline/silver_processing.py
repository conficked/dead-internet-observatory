"""
Silver → Gold processing pipeline.

Incremental: only scores documents not already present in the gold layer.
New docs are appended to scored.parquet; the durable registry and SQLite index
are updated with each run.
"""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from analytics.aliveness_index import AlivenessIndexEngine
from pipeline.id_reconciliation import atomic_parquet, id_database, load_ids, merge_registry

MAX_NEW_DOCS = 6_000
MAX_NEW_CC_DOCS = 50_000


def cc_gold_counts(gold_path):
    if not gold_path.exists():
        return {}
    counts = {}
    with pq.ParquetFile(gold_path) as parquet:
        for batch in parquet.iter_batches(columns=['source', 'crawl_partition'], batch_size=10_000):
            for source, partition in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if source == 'common_crawl' and partition is not None:
                    crawl = partition.replace('/', '-')
                    counts[crawl] = counts.get(crawl, 0) + 1
    return counts


def persist_cc_status(gold_path, conn):
    from datetime import datetime, timezone
    counts = cc_gold_counts(gold_path)
    conn.execute('CREATE TABLE IF NOT EXISTS cc_crawl_status (crawl_id TEXT PRIMARY KEY, gold_rows INTEGER, updated_at TEXT)')
    conn.execute('DELETE FROM cc_crawl_status')
    conn.executemany('INSERT INTO cc_crawl_status VALUES (?, ?, ?)',
                     [(cid, n, datetime.now(timezone.utc).isoformat()) for cid, n in counts.items()])


def count_unique_doc_ids(path):
    with id_database() as conn:
        load_ids(conn, 'ids', path)
        return conn.execute('SELECT COUNT(*) FROM ids').fetchone()[0]


def candidate_gold_ids(path, candidates):
    found = set()
    if path.exists():
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(columns=['doc_id'], batch_size=10_000):
                found.update(value for value in batch.column(0).to_pylist() if value in candidates)
    return found


def reconcile_counts(engine, seen_count, scored_count):
    with engine._conn() as conn:
        indexed_count = conn.execute("SELECT COUNT(*) FROM ingested_docs").fetchone()[0]
        counts = dict(total_seen_count=seen_count, total_scored_count=scored_count,
                      total_indexed_count=indexed_count)
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(key, str(value)) for key, value in counts.items()])
    if len(set(counts.values())) > 1:
        logging.getLogger(__name__).warning(
            "[GOLD] Reconciliation: seen=%d scored=%d indexed=%d",
            seen_count, scored_count, indexed_count)


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


def validate_delta(scored):
    if 'doc_id' not in scored or not scored['doc_id'].map(
            lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Every scored row must have a nonempty doc_id')
    return scored.drop_duplicates(subset=['doc_id'], keep='last').copy()


def select_silver_candidates(silver_path, gold_path, counts, *, max_cc=MAX_NEW_CC_DOCS,
                             per_crawl=10000, max_other=MAX_NEW_DOCS):
    with id_database() as conn:
        load_ids(conn, 'scored', gold_path)
        conn.execute('CREATE TABLE candidates (doc_id TEXT PRIMARY KEY, source TEXT, '
                     'crawl TEXT, created TEXT, sequence INTEGER, payload BLOB)')
        sequence = 0
        with pq.ParquetFile(silver_path) as parquet:
            columns = parquet.schema_arrow.names
            for batch in parquet.iter_batches(batch_size=1000):
                for row in batch.to_pylist():
                    sequence += 1
                    doc_id = row.get('doc_id')
                    if not isinstance(doc_id, str) or not doc_id.strip():
                        raise ValueError('Every silver row must have a nonempty doc_id')
                    created = pd.to_datetime(row.get('created_dt'), utc=True, errors='coerce')
                    conn.execute(
                        'INSERT OR REPLACE INTO candidates SELECT ?,?,?,?,?,? '
                        'WHERE NOT EXISTS (SELECT 1 FROM scored WHERE doc_id=?)',
                        (doc_id, row.get('source'), row.get('crawl_partition'),
                         None if pd.isna(created) else created.isoformat(), sequence,
                         pickle.dumps(row), doc_id))
                conn.commit()
        conn.execute('CREATE TABLE rooms (crawl TEXT PRIMARY KEY, room INTEGER)')
        conn.executemany('INSERT INTO rooms VALUES (?,?)',
                         ((crawl, max(0, per_crawl - count)) for crawl, count in counts.items()))
        cc = conn.execute(
            'WITH ranked AS (SELECT payload, crawl, ROW_NUMBER() OVER '
            '(PARTITION BY crawl ORDER BY sequence) AS round FROM candidates '
            "WHERE source='common_crawl' AND crawl IS NOT NULL) "
            'SELECT payload FROM ranked LEFT JOIN rooms '
            "ON rooms.crawl=REPLACE(ranked.crawl, '/', '-') "
            'WHERE round <= COALESCE(room, ?) ORDER BY round, ranked.crawl LIMIT ?',
            (per_crawl, max_cc))
        selected = [pickle.loads(row[0]) for row in cc]
        other = conn.execute(
            "SELECT payload FROM candidates WHERE source IS NULL OR source!='common_crawl' "
            'ORDER BY created DESC, sequence LIMIT ?', (max_other,))
        selected.extend(pickle.loads(row[0]) for row in other)
        return pd.DataFrame(selected, columns=columns)


def append_scored(gold_path, scored):
    """Merge the delta without materialising the historical corpus in pandas."""
    scored = validate_delta(scored)
    delta = pa.Table.from_pandas(scored, preserve_index=False)
    if not gold_path.exists():
        with atomic_parquet(gold_path) as temporary:
            pq.write_table(delta, temporary)
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

    def _recover_pending(self):
        pending = self.gold_root / "pending_delta.parquet"
        if not pending.exists():
            return
        scored = validate_delta(pd.read_parquet(pending))
        gold = self.gold_root / "scored.parquet"
        registry = self.gold_root / "doc_registry.parquet"
        append_scored(gold, scored)
        seen = merge_registry(registry, gold)
        self.engine.ingest_scored_delta(scored)
        reconcile_counts(self.engine, seen, count_unique_doc_ids(gold))
        with self.engine._conn() as conn:
            persist_cc_status(gold, conn)
        pending.unlink()

    def run(self, source_file: str = "combined.parquet"):
        self._recover_pending()
        from detection.ai_content_detector import score_dataframe, corpus_summary

        silver_path = self.silver_root / source_file
        if not silver_path.exists():
            print(f"[GOLD] Silver file not found: {silver_path}")
            return

        print(f"[GOLD] Loading {silver_path} …")

        gold_path = self.gold_root / "scored.parquet"
        registry_path = self.gold_root / "doc_registry.parquet"
        if not registry_path.exists():
            raise RuntimeError(
                "Missing gold/doc_registry.parquet; restore persistent state before scoring"
            )
        seen_count = count_unique_doc_ids(registry_path)
        scored_count = count_unique_doc_ids(gold_path) if gold_path.exists() else 0
        reconcile_counts(self.engine, seen_count, scored_count)
        cc_config = self.config.get('sources', {}).get('common_crawl', {})
        new_df = select_silver_candidates(
            silver_path, gold_path, cc_gold_counts(gold_path),
            max_cc=cc_config.get('max_new_docs_per_run', MAX_NEW_CC_DOCS),
            per_crawl=cc_config.get('max_rows_per_crawl', 10000))
        print(f"[GOLD] {len(new_df):,} new docs selected to score")

        if new_df.empty:
            seen_count = merge_registry(registry_path, gold_path)
            reconcile_counts(self.engine, seen_count, scored_count)
            print("[GOLD] Nothing new — index already up to date")
            with self.engine._conn() as conn:
                persist_cc_status(gold_path, conn)
            return

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

        scored = validate_delta(scored)
        pending = self.gold_root / "pending_delta.parquet"
        with atomic_parquet(pending) as temporary:
            scored.to_parquet(temporary, index=False, engine="pyarrow")
        self._recover_pending()
        print("[GOLD] ✓ Gold, registry and SQLite index updated")


if __name__ == "__main__":
    SilverToGoldPipeline().run()

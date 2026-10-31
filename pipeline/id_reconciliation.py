"""Disk-backed ID reconciliation; only Parquet batches enter Python memory."""

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@contextmanager
def id_database():
    with tempfile.TemporaryDirectory(prefix='observatory-ids-') as directory:
        conn = sqlite3.connect(str(Path(directory) / 'ids.db'))
        try:
            conn.execute('PRAGMA cache_size = -4096')
            conn.execute('PRAGMA temp_store = FILE')
            yield conn
        finally:
            conn.close()


def load_ids(conn, table, path):
    conn.execute(f'CREATE TABLE IF NOT EXISTS {table} (doc_id TEXT PRIMARY KEY) WITHOUT ROWID')
    if path is None or not Path(path).exists():
        return
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(columns=['doc_id'], batch_size=10_000):
            conn.executemany(f'INSERT OR IGNORE INTO {table} VALUES (?)',
                             ((value,) for value in batch.column(0).to_pylist() if value is not None))
    conn.commit()


@contextmanager
def atomic_parquet(destination):
    destination = Path(destination)
    fd, name = tempfile.mkstemp(prefix=f'.{destination.name}-', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def merge_registry(registry, gold):
    with id_database() as conn:
        load_ids(conn, 'ids', registry)
        load_ids(conn, 'ids', gold)
        count = conn.execute('SELECT COUNT(*) FROM ids').fetchone()[0]
        schema = pa.schema([('doc_id', pa.string())])
        with atomic_parquet(registry) as temporary:
            with pq.ParquetWriter(temporary, schema) as writer:
                cursor = conn.execute('SELECT doc_id FROM ids ORDER BY doc_id')
                while rows := cursor.fetchmany(10_000):
                    writer.write_table(pa.table({'doc_id': [row[0] for row in rows]}, schema=schema))
        return count

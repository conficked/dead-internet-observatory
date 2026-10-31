"""Read-only reconciliation of the registry, scored gold, and SQLite ledger."""

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.id_reconciliation import id_database, load_ids


DIVERGENCE_KEYS = (
    'registry_only_ids', 'scored_not_registered_ids', 'scored_not_indexed_ids',
    'indexed_not_scored_ids', 'null_dates', 'future_dated_indexed_rows',
)


def audit_state(data_root, db_path):
    data_root = Path(data_root)
    database = Path(db_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Active WAL state cannot be audited without a checkpointed snapshot')
    # Immutable reads prevent SQLite from creating WAL or shared-memory files.
    # Refuse pending WAL above rather than silently ignoring its observations.
    uri = database.as_uri() + '?mode=ro&immutable=1'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute('PRAGMA query_only = ON')
        null_dates = conn.execute(
            "SELECT COUNT(*) FROM ingested_docs WHERE date IS NULL OR trim(date) = ''"
        ).fetchone()[0]
        future = conn.execute(
            'SELECT COUNT(*) FROM ingested_docs WHERE date > ?',
            (datetime.now(timezone.utc).date().isoformat(),)
        ).fetchone()[0]
        with id_database() as ids:
            for name in ('doc_registry.parquet', 'scored.parquet'):
                if not (data_root / 'gold' / name).is_file():
                    raise FileNotFoundError(data_root / 'gold' / name)
            load_ids(ids, 'registry', data_root / 'gold/doc_registry.parquet')
            load_ids(ids, 'scored', data_root / 'gold/scored.parquet')
            load_ids(ids, 'indexed', None)
            ids.executemany('INSERT OR IGNORE INTO indexed VALUES (?)',
                            conn.execute('SELECT doc_id FROM ingested_docs WHERE doc_id IS NOT NULL'))
            def count(table):
                return ids.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            def difference(left, right):
                return ids.execute(
                    f'SELECT COUNT(*) FROM {left} a WHERE NOT EXISTS '
                    f'(SELECT 1 FROM {right} b WHERE b.doc_id = a.doc_id)').fetchone()[0]
            return {
                'registry_unique_ids': count('registry'),
                'scored_unique_ids': count('scored'),
                'indexed_ids': count('indexed'),
                'registry_only_ids': difference('registry', 'scored'),
                'scored_not_registered_ids': difference('scored', 'registry'),
                'scored_not_indexed_ids': difference('scored', 'indexed'),
                'indexed_not_scored_ids': difference('indexed', 'scored'),
                'null_dates': null_dates,
                'future_dated_indexed_rows': future,
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--db-path', type=Path)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = audit_state(args.data_root, args.db_path or args.data_root / 'observatory.db')
    except (OSError, ValueError, KeyError, sqlite3.Error, pa.ArrowException) as exc:
        report = {'error': str(exc)}
        status = 2
    else:
        status = int(any(report[key] for key in DIVERGENCE_KEYS))
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print({0: 'Pipeline state consistent', 1: 'Pipeline state divergence',
               2: 'Pipeline state unavailable'}[status])
        for key, value in report.items():
            print(f'{key}: {value}')
    return status


if __name__ == '__main__':
    raise SystemExit(main())

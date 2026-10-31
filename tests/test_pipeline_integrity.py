import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from analytics.aliveness_index import AlivenessIndexEngine
from pipeline import bronze_ingestion as bronze
from pipeline.silver_processing import SilverToGoldPipeline


AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_pipeline_state.py'


def audit_fixture(root, registry=('a',), scored=('a',), indexed=(('a', '2026-01-01'),)):
    gold = root / 'gold'
    gold.mkdir()
    pd.DataFrame({'doc_id': list(registry)}).to_parquet(gold / 'doc_registry.parquet')
    pd.DataFrame({'doc_id': list(scored), 'text': 'unused'}).to_parquet(gold / 'scored.parquet')
    with sqlite3.connect(root / 'observatory.db') as conn:
        conn.execute('CREATE TABLE ingested_docs (doc_id TEXT PRIMARY KEY, date TEXT)')
        conn.executemany('INSERT INTO ingested_docs VALUES (?, ?)', indexed)


def run_audit(root, *args):
    return subprocess.run([sys.executable, str(AUDIT_SCRIPT), '--data-root', str(root), *args],
                          capture_output=True, text=True)


def test_audit_consistent_and_read_only(tmp_path):
    audit_fixture(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    result = run_audit(tmp_path, '--json')
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['registry_unique_ids'] == report['scored_unique_ids'] == report['indexed_ids'] == 1
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()} == before
    assert 'consistent' in run_audit(tmp_path).stdout.lower()


def test_audit_divergent_state(tmp_path):
    audit_fixture(tmp_path, registry=('a', 'r'), scored=('a', 's'),
                  indexed=(('a', None), ('i', '2999-01-01')))
    result = run_audit(tmp_path, '--json')
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    for key in ('registry_only_ids', 'scored_not_registered_ids', 'scored_not_indexed_ids',
                'indexed_not_scored_ids', 'null_dates', 'future_dated_indexed_rows'):
        assert report[key] == 1


def test_audit_handles_non_value_arrow_errors(tmp_path, monkeypatch, capsys):
    import pyarrow as pa
    from scripts import audit_pipeline_state

    def fail(*args, **kwargs):
        raise pa.ArrowException('unreadable parquet')

    monkeypatch.setattr(audit_pipeline_state, 'load_ids', fail)
    audit_fixture(tmp_path)
    status = audit_pipeline_state.main(['--data-root', str(tmp_path), '--json'])
    assert status == 2
    assert 'unreadable parquet' in json.loads(capsys.readouterr().out)['error']


@pytest.mark.parametrize('corrupt', [False, True])
def test_audit_missing_or_corrupt_state(tmp_path, corrupt):
    if corrupt:
        audit_fixture(tmp_path)
        (tmp_path / 'gold/scored.parquet').write_bytes(b'corrupt')
    result = run_audit(tmp_path, '--json')
    assert result.returncode == 2
    assert 'error' in json.loads(result.stdout)
    if not corrupt:
        assert not (tmp_path / 'observatory.db').exists()


def test_audit_reads_only_required_gold_columns(tmp_path, monkeypatch):
    from scripts.audit_pipeline_state import audit_state
    audit_fixture(tmp_path)
    import pyarrow.parquet as pq
    original = pq.ParquetFile.iter_batches
    calls = []

    def read(self, **kwargs):
        calls.append(kwargs.get('columns'))
        yield from original(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', read)
    report = audit_state(tmp_path, tmp_path / 'observatory.db')
    assert report['scored_unique_ids'] == 1
    assert calls == [['doc_id'], ['doc_id']]


@pytest.fixture
def bronze_pipeline(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'storage': {'local_path': str(tmp_path),
        'bronze_path': 'bronze', 'silver_path': 'silver'}}))
    return bronze.BronzeToSilverPipeline(str(config))


def write_news(pipeline, records):
    path = pipeline.bronze_root / 'news' / '2026' / 'records.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(records)
    return path


def test_bronze_reports_all_ingestion_outcomes(bronze_pipeline, capsys):
    path = write_news(bronze_pipeline, '\n{bad json\n' + '\n'.join(json.dumps(x) for x in [
        {'text': 123}, {'text': ''}, {'text': 'accepted news'},
        {'text': 'accepted news'}, {'text': 'other news'}]) + '\n')
    df = bronze_pipeline.process_source('news')
    assert len(df) == 2
    assert bronze_pipeline.source_summaries['news'] == {
        'files_read': 1, 'lines_read': 7, 'blank_lines': 1,
        'malformed_json_lines': 1, 'normalization_failures': 1,
        'empty_or_unsupported_records': 1, 'duplicate_content_hashes': 1,
        'accepted_documents': 2, 'rejected_records': 3,
    }
    output = capsys.readouterr().out
    assert str(path) in output and 'line=3' in output and 'source=news' in output
    assert 'malformed_json_lines=1' in output and 'accepted_documents=2' in output


def test_bronze_reader_retains_two_value_interface(bronze_pipeline):
    write_news(bronze_pipeline, '{"text":"news"}\n')
    assert list(bronze.iter_bronze_records(bronze_pipeline.bronze_root, 'news')) == [
        ({'text': 'news'}, '2026')]


@pytest.mark.parametrize('failure,output', [
    ('write', 'news.parquet'), ('replace', 'news.parquet'),
    ('write', 'combined.parquet'), ('replace', 'combined.parquet'),
    ('close', 'combined.parquet'),
])
def test_silver_failure_preserves_output_and_cleans_temporary_files(
        bronze_pipeline, monkeypatch, output, failure):
    write_news(bronze_pipeline, '{"text":"news"}\n')
    target = bronze_pipeline.silver_root / output
    pd.DataFrame({'prior': [1]}).to_parquet(target)
    prior = target.read_bytes()
    if failure == 'replace':
        real_replace = type(target).replace

        def fail_replace(self, destination):
            if destination == target:
                raise OSError('replacement failed')
            return real_replace(self, destination)
        monkeypatch.setattr(type(target), 'replace', fail_replace)
    elif output == 'news.parquet':
        def fail_write(self, path, **kwargs):
            path.write_bytes(b'partial')
            raise OSError('write failed')
        monkeypatch.setattr(pd.DataFrame, 'to_parquet', fail_write)
    else:
        real_writer = bronze.pq.ParquetWriter

        class FailingWriter:
            def __init__(self, *args, **kwargs):
                self.writer = real_writer(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

            def write_table(self, table, *args, **kwargs):
                self.writer.write_table(table, *args, **kwargs)
                if failure == 'write':
                    raise OSError('write failed')

            def close(self):
                self.writer.close()
                if failure == 'close':
                    raise OSError('close failed')

        monkeypatch.setattr(bronze.pq, 'ParquetWriter', FailingWriter)
    with pytest.raises(OSError):
        if output == 'news.parquet':
            bronze_pipeline.process_source('news')
        else:
            bronze_pipeline.run_all()
    assert target.read_bytes() == prior
    assert not list(bronze_pipeline.silver_root.glob('*.tmp'))


@pytest.fixture
def integrity_engine(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'storage': {'db_path': str(tmp_path / 'index.db')}, 'analytics': {}}))
    return AlivenessIndexEngine(str(config))


def test_registry_only_ids_are_scored_and_counts_reconcile(tmp_path, monkeypatch, integrity_engine, caplog):
    pipeline = SilverToGoldPipeline.__new__(SilverToGoldPipeline)
    pipeline.config = {}
    pipeline.engine = integrity_engine
    pipeline.gold_root = tmp_path / 'gold'
    pipeline.silver_root = tmp_path / 'silver'
    pipeline.gold_root.mkdir()
    pipeline.silver_root.mkdir()
    frame = pd.DataFrame({'doc_id': ['replay'], 'source': ['reddit'], 'category': ['social'],
                          'crawl_partition': ['2026/01'], 'created_dt': ['2026-01-01']})
    frame.to_parquet(pipeline.silver_root / 'combined.parquet')
    pd.DataFrame({'doc_id': ['replay', 'ledger-only']}).to_parquet(pipeline.gold_root / 'doc_registry.parquet')
    calls = []
    def score(frame):
        calls.extend(frame.doc_id)
        return frame.assign(aliveness_score=75)
    monkeypatch.setitem(sys.modules, 'detection.ai_content_detector', SimpleNamespace(
        score_dataframe=score, corpus_summary=lambda frame: {}))
    with caplog.at_level(logging.WARNING):
        pipeline.run()
        pipeline.run()
    assert calls == ['replay']
    assert pd.read_parquet(pipeline.gold_root / 'scored.parquet').doc_id.tolist() == ['replay']
    assert integrity_engine.get_meta('total_seen_count') == '2'
    assert integrity_engine.get_meta('total_scored_count') == '1'
    assert integrity_engine.get_meta('total_indexed_count') == '1'
    assert 'reconciliation' in caplog.text.lower()
    with integrity_engine._conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM ingested_docs').fetchone()[0] == 1


def test_total_scored_count_recounts_unique_ids_from_gold(tmp_path, monkeypatch, integrity_engine):
    pipeline = SilverToGoldPipeline.__new__(SilverToGoldPipeline)
    pipeline.config = {}
    pipeline.engine = integrity_engine
    pipeline.gold_root = tmp_path / 'gold'
    pipeline.silver_root = tmp_path / 'silver'
    pipeline.gold_root.mkdir()
    pipeline.silver_root.mkdir()
    old = pd.DataFrame({
        'doc_id': ['old', 'old'], 'source': ['reddit', 'reddit'],
        'category': ['social', 'social'], 'crawl_partition': ['2026/01', '2026/01'],
        'created_dt': pd.to_datetime(['2026-01-01', '2026-01-01'], utc=True),
        'aliveness_score': [60, 60],
    })
    old.to_parquet(pipeline.gold_root / 'scored.parquet')
    pd.DataFrame({'doc_id': ['old']}).to_parquet(pipeline.gold_root / 'doc_registry.parquet')
    pd.DataFrame({
        'doc_id': ['new'], 'source': ['reddit'], 'category': ['social'],
        'crawl_partition': ['2026/01'],
        'created_dt': pd.to_datetime(['2026-01-02'], utc=True),
    }).to_parquet(pipeline.silver_root / 'combined.parquet')
    monkeypatch.setitem(sys.modules, 'detection.ai_content_detector', SimpleNamespace(
        score_dataframe=lambda frame: frame.assign(aliveness_score=75),
        corpus_summary=lambda frame: {}))

    pipeline.run()

    assert len(pd.read_parquet(pipeline.gold_root / 'scored.parquet')) == 3
    assert integrity_engine.get_meta('total_scored_count') == '2'


def test_indexed_metadata_tracks_direct_ingestion(integrity_engine):
    frame = pd.DataFrame({'doc_id': ['a'], 'created_dt': ['2026-01-01'], 'source': ['reddit'],
                          'category': ['social'], 'aliveness_score': [50]})
    integrity_engine.ingest_scored_delta(frame)
    integrity_engine.ingest_scored_delta(frame)
    assert integrity_engine.get_meta('total_indexed_count') == '1'


def test_timestamp_rejections_are_separate_and_have_no_grace(integrity_engine, monkeypatch, caplog):
    from datetime import datetime, timezone
    import analytics.aliveness_index as index
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(index, 'datetime', FixedDatetime)
    frame = pd.DataFrame({'doc_id': ['valid', 'near', 'far', 'null', 'invalid'],
                          'created_dt': ['2026-10-31T12:00:00Z', '2026-10-31T12:00:01Z',
                                         '2027-01-01', None, 'invalid'],
                          'source': 'reddit', 'category': 'social', 'aliveness_score': 50})
    with caplog.at_level(logging.INFO):
        integrity_engine.ingest_scored_delta(frame)
    with integrity_engine._conn() as conn:
        assert [row[0] for row in conn.execute('SELECT doc_id FROM ingested_docs')] == ['valid']
    assert '1 null' in caplog.text
    assert '1 invalid' in caplog.text
    assert '2 future' in caplog.text


def test_current_score_ignores_future_and_reconciliation_preserves_rows(integrity_engine, monkeypatch):
    from datetime import datetime, timezone
    import analytics.aliveness_index as index
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(index, 'datetime', FixedDatetime)
    with integrity_engine._conn() as conn:
        conn.executemany('INSERT INTO composite_index VALUES (?,?,?,?,?,?)',
                         [('2026-10-31', 40, 40, 10, 0, ''), ('2026-11-01', 90, 90, 10, 0, '')])
        conn.execute('INSERT INTO ingested_docs VALUES (?,?,?,?,?,?)',
                     ('future', '2026-11-01', 'reddit', 'social', 90, None))
    assert integrity_engine.get_current_score() == 40
    assert integrity_engine.reconcile_future_dates() == {'ingested_docs': 1, 'daily_index': 0,
                                                         'composite_index': 1, 'domain_scores': 0}
    with integrity_engine._conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM composite_index').fetchone()[0] == 2
        assert conn.execute('SELECT COUNT(*) FROM ingested_docs').fetchone()[0] == 1


def test_audit_checkpointed_wal_creates_no_sidecar_files(tmp_path):
    audit_fixture(tmp_path)
    with sqlite3.connect(tmp_path / 'observatory.db') as conn:
        conn.execute('PRAGMA journal_mode=WAL')
    conn.close()
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = run_audit(tmp_path, '--json')
    assert result.returncode == 0, result.stdout
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


def test_audit_refuses_uncheckpointed_wal_without_modification(tmp_path):
    audit_fixture(tmp_path)
    conn = sqlite3.connect(tmp_path / 'observatory.db')
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute("INSERT INTO ingested_docs VALUES ('pending', '2026-01-01')")
        conn.commit()
        before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        result = run_audit(tmp_path, '--json')
        assert result.returncode == 2
        assert 'WAL' in json.loads(result.stdout)['error']
        assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    finally:
        conn.close()


@pytest.mark.parametrize('destination', ['scored.parquet', 'doc_registry.parquet'])
def test_pending_delta_recovers_after_replacement(tmp_path, monkeypatch, integrity_engine, destination):
    from pipeline import silver_processing as gold
    pipeline = SilverToGoldPipeline.__new__(SilverToGoldPipeline)
    pipeline.config = {}
    pipeline.engine = integrity_engine
    pipeline.gold_root = tmp_path / 'gold'
    pipeline.silver_root = tmp_path / 'silver'
    pipeline.gold_root.mkdir()
    pipeline.silver_root.mkdir()
    pd.DataFrame({'doc_id': []}).to_parquet(pipeline.gold_root / 'doc_registry.parquet')
    pd.DataFrame({'doc_id': ['new'], 'source': ['reddit'], 'category': ['social'],
                  'crawl_partition': ['2026/01'], 'created_dt': ['2026-01-01']}).to_parquet(
                      pipeline.silver_root / 'combined.parquet')
    calls = []
    def score(frame):
        calls.extend(frame.doc_id)
        return frame.assign(aliveness_score=75)
    monkeypatch.setitem(sys.modules, 'detection.ai_content_detector', SimpleNamespace(
        score_dataframe=score, corpus_summary=lambda frame: {}))
    original = gold.os.replace
    def fail_after_replace(src, dst):
        original(src, dst)
        if Path(dst).name == destination:
            raise OSError('crash after replacement')
    with monkeypatch.context() as patch:
        patch.setattr(gold.os, 'replace', fail_after_replace)
        with pytest.raises(OSError, match='crash'):
            pipeline.run()
    assert (pipeline.gold_root / 'pending_delta.parquet').exists()
    # Recovery must run even when silver is unavailable.
    (pipeline.silver_root / 'combined.parquet').unlink()
    pipeline.run()
    pipeline.run()
    assert calls == ['new']
    assert pd.read_parquet(pipeline.gold_root / 'scored.parquet').doc_id.tolist() == ['new']
    assert pd.read_parquet(pipeline.gold_root / 'doc_registry.parquet').doc_id.tolist() == ['new']
    with integrity_engine._conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM ingested_docs').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM cc_crawl_status').fetchone()[0] == 0
    for key in ('total_seen_count', 'total_scored_count', 'total_indexed_count'):
        assert integrity_engine.get_meta(key) == '1'
    assert sorted(p.name for p in pipeline.gold_root.iterdir()) == ['doc_registry.parquet', 'scored.parquet']


@pytest.mark.parametrize('failure', ['later_source', 'publication'])
def test_run_all_preserves_entire_silver_group(bronze_pipeline, monkeypatch, failure):
    write_news(bronze_pipeline, '{"text":"new news"}\n')
    root = bronze_pipeline.silver_root
    for name in ('common_crawl.parquet', 'news.parquet', 'combined.parquet'):
        pd.DataFrame({'old': [name]}).to_parquet(root / name)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    if failure == 'later_source':
        original = bronze_pipeline.process_source
        def process(source):
            if source == 'wayback':
                raise OSError('later source failed')
            return original(source)
        monkeypatch.setattr(bronze_pipeline, 'process_source', process)
    else:
        original = Path.replace
        def replace(self, destination):
            if destination == root / 'combined.parquet':
                raise OSError('publication failed')
            return original(self, destination)
        monkeypatch.setattr(Path, 'replace', replace)
    with pytest.raises(OSError):
        bronze_pipeline.run_all()
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_id_reconciliation_uses_batches(tmp_path, monkeypatch):
    from pipeline.silver_processing import count_unique_doc_ids
    from scripts.audit_pipeline_state import audit_state
    import pyarrow.parquet as pq
    audit_fixture(tmp_path, registry=('a', 'a'), scored=('a', 'a'))
    original = pq.ParquetFile.iter_batches
    calls = []
    def batches(self, *args, **kwargs):
        calls.append(kwargs)
        yield from original(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', batches)
    def reject(*args, **kwargs):
        pytest.fail('full pandas ID read')
    monkeypatch.setattr(pd, 'read_parquet', reject)
    assert count_unique_doc_ids(tmp_path / 'gold/scored.parquet') == 1
    assert audit_state(tmp_path, tmp_path / 'observatory.db')['scored_unique_ids'] == 1
    assert len(calls) >= 3
    assert all(call['columns'] == ['doc_id'] and call['batch_size'] <= 100_000 for call in calls)


@pytest.mark.parametrize('failure', ['ingest', 'counts', 'status'])
def test_pending_delta_survives_downstream_failure(tmp_path, monkeypatch, integrity_engine, failure):
    from pipeline import silver_processing as gold
    pipeline = SilverToGoldPipeline.__new__(SilverToGoldPipeline)
    pipeline.engine = integrity_engine
    pipeline.gold_root = tmp_path
    frame = pd.DataFrame({'doc_id': ['a'], 'source': ['common_crawl'], 'category': ['web'],
                          'crawl_partition': ['2026/01'], 'created_dt': ['2026-01-01'],
                          'aliveness_score': [70]})
    pending = tmp_path / 'pending_delta.parquet'
    frame.to_parquet(pending)
    def fail(*args, **kwargs):
        raise OSError('interrupted')
    with monkeypatch.context() as patch:
        if failure == 'ingest':
            patch.setattr(integrity_engine, 'ingest_scored_delta', fail)
        else:
            patch.setattr(gold, 'reconcile_counts' if failure == 'counts' else 'persist_cc_status', fail)
        with pytest.raises(OSError):
            pipeline._recover_pending()
    assert pending.exists()
    pipeline._recover_pending()
    pipeline._recover_pending()
    assert not pending.exists()
    assert pd.read_parquet(tmp_path / 'scored.parquet').doc_id.tolist() == ['a']
    with integrity_engine._conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM ingested_docs').fetchone()[0] == 1
        assert conn.execute('SELECT gold_rows FROM cc_crawl_status').fetchone()[0] == 1
        assert conn.execute('SELECT n_docs FROM daily_index').fetchone()[0] == 1
    assert integrity_engine.get_meta('total_scored_count') == '1'


def test_registry_merge_is_batched_atomic_and_cleans_disk_state(tmp_path, monkeypatch):
    from pipeline import id_reconciliation as ids
    registry = tmp_path / 'registry.parquet'
    gold = tmp_path / 'gold.parquet'
    pd.DataFrame({'doc_id': ['a', 'a', None]}).to_parquet(registry)
    pd.DataFrame({'doc_id': ['b', 'a', None]}).to_parquet(gold)
    original = ids.sqlite3.connect
    databases = []
    def connect(path, *args, **kwargs):
        databases.append(Path(path))
        assert str(path) != ':memory:'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(ids.sqlite3, 'connect', connect)
    def reject(*args, **kwargs):
        pytest.fail('full pandas ID read')
    with monkeypatch.context() as patch:
        patch.setattr(pd, 'read_parquet', reject)
        assert ids.merge_registry(registry, gold) == 2
    assert pd.read_parquet(registry).doc_id.tolist() == ['a', 'b']
    before = registry.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('replace failed')
    with monkeypatch.context() as patch:
        patch.setattr(ids.os, 'replace', fail)
        with pytest.raises(OSError):
            ids.merge_registry(registry, gold)
    assert registry.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['gold.parquet', 'registry.parquet']
    assert len(databases) == 2
    assert all(not path.parent.exists() for path in databases)


def test_candidate_membership_only_retains_candidates(tmp_path, monkeypatch):
    from pipeline.silver_processing import candidate_gold_ids
    path = tmp_path / 'gold.parquet'
    pd.DataFrame({'doc_id': ['old', 'candidate', None, 'other']}).to_parquet(path)
    def reject(*args, **kwargs):
        pytest.fail('full pandas ID read')
    monkeypatch.setattr(pd, 'read_parquet', reject)
    assert candidate_gold_ids(path, {'candidate', 'absent'}) == {'candidate'}


@pytest.mark.parametrize('bad', [None, '', '   '])
def test_invalid_pending_retries_leave_all_stores_unchanged(tmp_path, integrity_engine, bad):
    pipeline = SilverToGoldPipeline.__new__(SilverToGoldPipeline)
    pipeline.gold_root = tmp_path
    pipeline.engine = integrity_engine
    frame = pd.DataFrame({'doc_id': ['old'], 'source': ['reddit'], 'category': ['social'],
                          'created_dt': ['2026-01-01'], 'crawl_partition': ['2026/01'],
                          'aliveness_score': [50]})
    frame.to_parquet(tmp_path / 'scored.parquet')
    frame[['doc_id']].to_parquet(tmp_path / 'doc_registry.parquet')
    frame.assign(doc_id=bad).to_parquet(tmp_path / 'pending_delta.parquet')
    before = {p: p.read_bytes() for p in tmp_path.glob('*.parquet')}
    with integrity_engine._conn() as conn:
        database = list(conn.iterdump())
    for _ in range(2):
        with pytest.raises(ValueError, match='doc_id'):
            pipeline._recover_pending()
        assert {p: p.read_bytes() for p in tmp_path.glob('*.parquet')} == before
        with integrity_engine._conn() as conn:
            assert list(conn.iterdump()) == database


@pytest.mark.parametrize('crash_at', ['news.parquet', 'combined.parquet', 'committed'])
def test_silver_crash_rolls_back_on_startup(bronze_pipeline, crash_at):
    write_news(bronze_pipeline, '{"text":"new news"}\n')
    root = bronze_pipeline.silver_root
    for name in ('news.parquet', 'combined.parquet', 'reddit.parquet'):
        pd.DataFrame({'old': [name]}).to_parquet(root / name)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    script = '''
import os, sys
from pathlib import Path
from pipeline.bronze_ingestion import BronzeToSilverPipeline
pipeline = BronzeToSilverPipeline(sys.argv[1])
import pipeline.bronze_ingestion as bronze
original_manifest = bronze._write_manifest
def manifest(transaction, data):
    original_manifest(transaction, data)
    if data['state'] == sys.argv[2]:
        os._exit(71)
bronze._write_manifest = manifest
original = Path.replace
def replace(self, destination):
    result = original(self, destination)
    if Path(destination).parent == pipeline.silver_root and Path(destination).name == sys.argv[2]:
        os._exit(71)
    return result
Path.replace = replace
pipeline.run_all()
'''
    result = subprocess.run([sys.executable, '-c', script, str(root.parent / 'config.yaml'), crash_at])
    assert result.returncode == 71
    bronze.BronzeToSilverPipeline(str(root.parent / 'config.yaml'))
    if crash_at == 'committed':
        assert sorted(p.name for p in root.iterdir()) == ['combined.parquet', 'news.parquet']
        assert len(pd.read_parquet(root / 'combined.parquet')) == 1
    else:
        assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_silver_empty_source_removes_stale_output(bronze_pipeline):
    pd.DataFrame({'old': [1]}).to_parquet(bronze_pipeline.silver_root / 'reddit.parquet')
    bronze_pipeline.run_all()
    assert not (bronze_pipeline.silver_root / 'reddit.parquet').exists()
    assert sorted(p.name for p in bronze_pipeline.silver_root.iterdir()) == ['combined.parquet']


def test_streamed_candidate_window_is_bounded_and_deduplicated(tmp_path, monkeypatch):
    from pipeline.silver_processing import select_silver_candidates
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [dict(doc_id=f'{source}-{crawl}-{i}', source=source, crawl_partition=crawl,
                 created_dt=f'2026-01-{i + 1:02}')
            for source, crawl in [('common_crawl', 'a'), ('common_crawl', 'b'), ('reddit', 'x')]
            for i in range(5)]
    rows.append(dict(rows[-1], created_dt='2026-02-01'))
    silver = tmp_path / 'combined.parquet'
    pq.write_table(pa.Table.from_pylist(rows), silver, row_group_size=2)
    gold = tmp_path / 'scored.parquet'
    pd.DataFrame({'doc_id': ['reddit-x-3']}).to_parquet(gold)
    original = pq.ParquetFile.iter_batches
    sizes = []
    def batches(self, **kwargs):
        kwargs['batch_size'] = 2
        for batch in original(self, **kwargs):
            sizes.append(len(batch))
            yield batch
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', batches)
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: pytest.fail('full parquet read'))
    original_frame = pd.DataFrame
    def bounded_frame(data, **kwargs):
        assert len(data) <= 5, 'all silver rows materialized'
        return original_frame(data, **kwargs)
    monkeypatch.setattr(pd, 'DataFrame', bounded_frame)
    result = select_silver_candidates(silver, gold, {'a': 1}, max_cc=3, per_crawl=2, max_other=2)
    assert result.doc_id.tolist() == ['common_crawl-a-0', 'common_crawl-b-0', 'common_crawl-b-1', 'reddit-x-4', 'reddit-x-2']
    assert result.iloc[3].created_dt == '2026-02-01'
    assert max(sizes) == 2


@pytest.mark.parametrize('bad_output', [True, False])
def test_scoring_validates_and_deduplicates_before_pending_write(
        tmp_path, monkeypatch, integrity_engine, bad_output):
    pipeline = SilverToGoldPipeline.__new__(SilverToGoldPipeline)
    pipeline.config = {}
    pipeline.engine = integrity_engine
    pipeline.gold_root = tmp_path / 'gold'
    pipeline.silver_root = tmp_path / 'silver'
    pipeline.gold_root.mkdir()
    pipeline.silver_root.mkdir()
    pd.DataFrame({'doc_id': []}).to_parquet(pipeline.gold_root / 'doc_registry.parquet')
    frame = pd.DataFrame({'doc_id': ['a'], 'source': ['reddit'], 'category': ['social'],
                          'created_dt': ['2026-01-01'], 'crawl_partition': ['2026/01']})
    frame.to_parquet(pipeline.silver_root / 'combined.parquet')
    def score(frame):
        return pd.concat([frame.assign(aliveness_score=10),
                          frame.assign(doc_id=None if bad_output else 'a', aliveness_score=70)])
    monkeypatch.setitem(sys.modules, 'detection.ai_content_detector', SimpleNamespace(
        score_dataframe=score, corpus_summary=lambda frame: {}))
    original_write = pd.DataFrame.to_parquet
    written = []
    def write(self, path, **kwargs):
        if 'pending_delta' in str(path):
            written.append(self.copy())
        return original_write(self, path, **kwargs)
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', write)
    original_read = pd.read_parquet
    def read(path, **kwargs):
        assert Path(path).name != 'combined.parquet', 'full silver pandas read'
        return original_read(path, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', read)
    if bad_output:
        with pytest.raises(ValueError, match='doc_id'):
            pipeline.run()
        assert written == []
        assert not (pipeline.gold_root / 'scored.parquet').exists()
    else:
        pipeline.run()
        assert len(written) == 1
        assert written[0].doc_id.tolist() == ['a']
        assert written[0].aliveness_score.tolist() == [70]
        assert pd.read_parquet(pipeline.gold_root / 'scored.parquet').aliveness_score.tolist() == [70]
    assert not (pipeline.gold_root / 'pending_delta.parquet').exists()


@pytest.mark.parametrize('invalid', [False, True])
def test_candidate_ledger_cleans_tmpdir_on_success_and_failure(tmp_path, monkeypatch, invalid):
    from pipeline import id_reconciliation as ids
    from pipeline.silver_processing import select_silver_candidates
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    silver = tmp_path / 'combined.parquet'
    pd.DataFrame({'doc_id': [None if invalid else 'a'], 'source': ['reddit']}).to_parquet(silver)
    original = ids.sqlite3.connect
    databases = []
    def connect(path, *args, **kwargs):
        databases.append(Path(path))
        assert Path(path).parent.parent == tmp_path
        return original(path, *args, **kwargs)
    monkeypatch.setattr(ids.sqlite3, 'connect', connect)
    if invalid:
        with pytest.raises(ValueError, match='doc_id'):
            select_silver_candidates(silver, tmp_path / 'absent.parquet', {})
    else:
        assert select_silver_candidates(silver, tmp_path / 'absent.parquet', {}).doc_id.tolist() == ['a']
    assert len(databases) == 1
    assert not databases[0].parent.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['combined.parquet']

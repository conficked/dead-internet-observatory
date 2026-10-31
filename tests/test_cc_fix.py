import json
import logging
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest


def test_independent_quotas_round_robin_and_existing_cap():
    from pipeline.silver_processing import select_new_docs
    df = pd.DataFrame([dict(source=src, crawl_partition=cid, created_dt=f'2026-01-{i+1:02}', doc_id=f'{src}{cid}{i}')
                       for src, cid in [('common_crawl','2020/01'), ('common_crawl','2021/01'), ('reddit','')]
                       for i in range(5)])
    got = select_new_docs(df, {'2020-01': 9}, max_cc=4, per_crawl=10, max_other=2)
    assert got.source.value_counts().to_dict() == {'common_crawl': 4, 'reddit': 2}
    assert got[got.source.eq('common_crawl')].crawl_partition.tolist() == ['2020/01', '2021/01', '2021/01', '2021/01']
    assert got[got.source.eq('reddit')].created_dt.tolist() == ['2026-01-05','2026-01-04']


def test_counts_persisted_from_gold(tmp_path):
    from pipeline.silver_processing import persist_cc_status
    import sqlite3
    gold = tmp_path / 'scored.parquet'
    pd.DataFrame({'source':['common_crawl','common_crawl','reddit'], 'crawl_partition':['2020/01']*3, 'text':['ignore']*3}).to_parquet(gold)
    db = tmp_path / 'observatory.db'
    with sqlite3.connect(db) as conn:
        persist_cc_status(gold, conn)
        assert conn.execute('SELECT crawl_id, gold_rows FROM cc_crawl_status').fetchall() == [('2020-01',2)]


def bot(tmp_path, monkeypatch, counts):
    from minions.common_crawl_bot import CommonCrawlMinion
    b = CommonCrawlMinion.__new__(CommonCrawlMinion)
    b.config = {'sources':{'common_crawl':{'max_rows_per_crawl':10, 'max_dates_per_run':5}}}
    b.PROGRESS_FILE = str(tmp_path/'progress.json'); b.ATTEMPTS_FILE = str(tmp_path/'attempts.json')
    b.logger = logging.getLogger('test'); b.stats = dict(fetched=0,processed=0,skipped=0,errors=0)
    b.max_segments = 2; b.max_per_wet = 5
    b._all_known_crawl_ids = lambda: ['2020-01']
    b._published_counts = lambda: counts
    b.get_wet_paths = lambda _: ['a','b','c']
    b.stream_wet_records = lambda _: iter([{'text':'ok'}]*5)
    b.save_bronze = Mock(); b.throttle = Mock(); b.report_stats = Mock()
    monkeypatch.setattr('minions.common_crawl_bot.time.sleep', lambda _: None)
    return b


def test_attempt_is_not_completion_and_cooldown(tmp_path, monkeypatch):
    b = bot(tmp_path,monkeypatch,{})
    b.run()
    assert json.loads(Path(b.PROGRESS_FILE).read_text()) == []
    assert json.loads(Path(b.ATTEMPTS_FILE).read_text())['2020-01']['segments_ok'] == 2
    assert b.stats['fetched'] == 10
    b.run(); assert b.stats['fetched'] == 10
    b._published_counts = lambda: {'2020-01':8}
    b.run(); assert json.loads(Path(b.PROGRESS_FILE).read_text()) == ['2020-01']
    b.run(); assert b.stats['fetched'] == 10


def test_segment_failures_retry_and_replace(tmp_path, monkeypatch):
    b = bot(tmp_path,monkeypatch,None)
    calls = []
    def stream(path):
        calls.append(path)
        if path == 'a': raise RuntimeError('503')
        return iter([{}]*5)
    monkeypatch.setattr('minions.common_crawl_bot.random.sample',lambda paths,n:list(paths))
    b.stream_wet_records = stream
    b.run()
    assert calls.count('a') == 4
    assert b.stats['fetched'] == 10
    assert json.loads(Path(b.PROGRESS_FILE).read_text()) == []


def test_status_fetch_retries_and_reads_sqlite(tmp_path, monkeypatch):
    import sqlite3
    from minions import common_crawl_bot as cc
    db = tmp_path/'published.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE cc_crawl_status (crawl_id TEXT, gold_rows INTEGER)')
        conn.execute("INSERT INTO cc_crawl_status VALUES ('2020-01', 8000)")
    b = bot(tmp_path, monkeypatch, None)
    response = Mock(content=db.read_bytes())
    get = Mock(side_effect=[RuntimeError('503'), response])
    monkeypatch.setattr(cc.requests, 'get', get)
    assert cc.CommonCrawlMinion._published_counts(b) == {'2020-01':8000}
    assert get.call_count == 2
    assert get.call_args.args[0] == cc.CommonCrawlMinion.STATUS_URL


def test_small_target_requires_every_file_untruncated(tmp_path, monkeypatch):
    b = bot(tmp_path, monkeypatch, {})
    b.max_segments = 3
    b.get_wet_paths = lambda _: ['a','b']
    b.stream_wet_records = lambda path: iter([{}] * (5 if path == 'a' else 1))
    b.run()
    assert 'available_records' not in json.loads(Path(b.ATTEMPTS_FILE).read_text())['2020-01']


def test_genuinely_exhausted_small_crawl_uses_smaller_target(tmp_path, monkeypatch):
    b = bot(tmp_path,monkeypatch,{})
    b.get_wet_paths = lambda _: ['a']
    b.stream_wet_records = lambda _: iter([{}]*3)
    b.run()
    b._published_counts = lambda: {'2020-01':3}
    b.run()
    assert b._load_progress() == {'2020-01'}


def test_dry_run_does_not_write_state(tmp_path, monkeypatch):
    b = bot(tmp_path,monkeypatch,{'2020-01':10})
    b.run(dry_run=True)
    assert not Path(b.PROGRESS_FILE).exists()
    assert not Path(b.ATTEMPTS_FILE).exists()


def test_pipeline_refreshes_status_when_every_cc_candidate_is_capped(tmp_path, monkeypatch):
    import sqlite3
    import sys
    from types import SimpleNamespace
    from pipeline.silver_processing import SilverToGoldPipeline
    fake_detector = SimpleNamespace(score_dataframe=Mock(), corpus_summary=Mock())
    monkeypatch.setitem(sys.modules, 'detection.ai_content_detector', fake_detector)
    b = SilverToGoldPipeline.__new__(SilverToGoldPipeline)
    b.config = {'sources':{'common_crawl':{'max_rows_per_crawl':2}}}
    b.silver_root = tmp_path/'silver'; b.silver_root.mkdir()
    b.gold_root = tmp_path/'gold'; b.gold_root.mkdir()
    pd.DataFrame({'doc_id':['old1','old2'],'source':['common_crawl']*2,'crawl_partition':['2020/01']*2}).to_parquet(b.gold_root/'scored.parquet')
    pd.DataFrame({'doc_id':['old1','old2']}).to_parquet(b.gold_root/'doc_registry.parquet')
    pd.DataFrame({'doc_id':['new'],'source':['common_crawl'],'crawl_partition':['2020/01']}).to_parquet(b.silver_root/'combined.parquet')
    db = tmp_path/'observatory.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)')
        conn.execute('CREATE TABLE ingested_docs (doc_id TEXT PRIMARY KEY)')
    b.engine = SimpleNamespace(_conn=lambda: sqlite3.connect(db))
    b.run()
    fake_detector.score_dataframe.assert_not_called()
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT crawl_id,gold_rows FROM cc_crawl_status').fetchall() == [('2020-01',2)]


def test_missing_wet_index_is_recorded_as_failed_attempt(tmp_path, monkeypatch):
    b = bot(tmp_path, monkeypatch, None)
    b.get_wet_paths = Mock(return_value=[])
    b.run()
    attempt = json.loads(Path(b.ATTEMPTS_FILE).read_text())['2020-01']
    assert attempt['segments_ok'] == 0 and attempt['records'] == 0
    b.run()
    assert b.get_wet_paths.call_count == 1
    assert b._load_progress() == set()

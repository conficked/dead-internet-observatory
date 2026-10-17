import logging

import pandas as pd
import pytest
import yaml

from analytics.aliveness_index import AlivenessIndexEngine
from pipeline import bronze_ingestion as bronze
from minions.base_minion import BaseMinion


@pytest.mark.parametrize('normalise', list(bronze.NORMALISERS.values()))
@pytest.mark.parametrize('raw_hash', [None, ''])
def test_missing_hash_uses_normalised_text(normalise, raw_hash):
    record = normalise({'text': '  distinct human text  ', 'content_hash': raw_hash,
                        'type': 'article_sample', 'comment_text': '  distinct human text  '}, '2026/01')
    if isinstance(record, list):
        record = record[0]
    assert record['content_hash'] == BaseMinion.content_hash('distinct human text')


@pytest.fixture
def engine(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'storage': {'db_path': str(tmp_path / 'index.db')},
                                     'analytics': {}, 'app': {'current_aliveness': 50}}))
    return AlivenessIndexEngine(str(config))


def docs(scores, ids=None):
    return pd.DataFrame({'doc_id': ids or [str(i) for i in range(len(scores))],
                         'source': 'reddit', 'category': 'social',
                         'created_dt': '2026-01-01', 'aliveness_score': scores})


@pytest.mark.parametrize('method', ['ingest_scored_df', 'ingest_scored_delta'])
def test_invalid_dates_are_dropped_and_logged(engine, method, caplog):
    df = docs([10, 20, 30, 40])
    df['created_dt'] = ['2026-01-01', None, '2999-01-01', 'not a date']
    with caplog.at_level(logging.INFO):
        getattr(engine, method)(df)
    assert engine.get_total_docs() == 1
    assert 'null' in caplog.text and 'future' in caplog.text


def test_delta_is_idempotent_and_recomputes_exact_statistics(engine):
    first = docs([10, 20, 90], ['a', 'b', 'c'])
    engine.ingest_scored_df(first)
    delta = docs([90, 100, 100], ['c', 'd', 'e'])
    engine.ingest_scored_delta(delta)
    engine.ingest_scored_delta(delta)
    row = engine.get_source_breakdown().iloc[0]
    expected = pd.Series([10, 20, 90, 100, 100])
    assert row.n_docs == 5
    assert row.median_score == 90
    assert row.mean_score == expected.mean()
    assert row.std_score == round(expected.std(), 3)
    assert row.pct_below_50 == 40


def test_synthetic_seeder_is_not_exposed():
    import analytics.aliveness_index as index
    import run_minions
    assert not hasattr(index, 'seed_' + 'demo_data')
    assert 'seed-' + 'demo' not in run_minions.COMMANDS


def test_no_current_score_without_observations(engine):
    assert engine.get_current_score() is None


def test_group_replacement_removes_obsolete_domain_scores(engine):
    first = docs([10], ['a'])
    first['domain'] = 'old.example'
    engine.ingest_scored_df(first)
    replacement = docs([80], ['b'])
    replacement['domain'] = 'new.example'
    engine.ingest_scored_df(replacement)
    assert engine.get_top_domains()['domain'].tolist() == ['new.example']


def test_legacy_schema_delta_merges_approximately_and_deduplicates(engine, caplog):
    with engine._conn() as conn:
        conn.execute('DROP TABLE ingested_docs')
        conn.execute('INSERT INTO daily_index VALUES (?,?,?,?,?,?,?,?,?,?)',
                     ('2026-01-01', 'reddit', 'social', 10, 20, 20, 1, 100, 0, 20))
    engine._init_db()
    delta = docs([50, 60, 100], ['fresh', 'a', 'b'])
    delta.loc[0, 'source'] = 'news'
    with caplog.at_level(logging.WARNING):
        engine.ingest_scored_delta(delta)
    row = engine.get_source_breakdown().set_index('source').loc['reddit']
    assert row.n_docs == 12
    assert row.mean_score == round((10 * 20 + 60 + 100) / 12, 3)
    assert row.median_score == 30  # Weighted old and delta medians.
    assert row.pct_below_50 == 83.33
    expected = pd.Series([20] * 10 + [60, 100])
    variance = (9 + ((expected - expected.mean()) ** 2).sum()) / 11
    assert row.std_score == round(variance ** 0.5, 3)
    assert row.aliveness_index == row.mean_score
    assert '1 legacy' in caplog.text and 'approximately' in caplog.text
    before = engine.get_source_breakdown()
    caplog.clear()
    engine.ingest_scored_delta(delta)
    pd.testing.assert_frame_equal(engine.get_source_breakdown(), before)
    assert not caplog.records
    # A later delta must retain the aggregate-only history, even after restart.
    engine = AlivenessIndexEngine(str(engine.db_path).replace('index.db', 'config.yaml'))
    engine.ingest_scored_delta(docs([60, 40], ['a', 'c']))
    row = engine.get_source_breakdown().set_index('source').loc['reddit']
    assert row.n_docs == 13
    assert row.mean_score == round((12 * before.set_index('source').loc['reddit', 'mean_score'] + 40) / 13, 3)
    with engine._conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM ingested_docs').fetchone()[0] == 4


def test_domain_aggregates_remain_exact_across_groups_and_reingestion(engine):
    first = docs([10, 90], ['a', 'b'])
    first['domain'] = 'shared.example'
    first.loc[1, 'category'] = 'forum'
    engine.ingest_scored_df(first)
    replacement = docs([50], ['c'])
    replacement['domain'] = 'new.example'
    engine.ingest_scored_df(replacement)
    engine.ingest_scored_delta(first.iloc[1:])
    domains = engine.get_top_domains().set_index('domain')
    assert domains.loc['shared.example', 'n_docs'] == 1
    assert domains.loc['shared.example', 'mean_score'] == 90

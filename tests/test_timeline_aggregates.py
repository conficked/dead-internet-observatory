import pandas as pd
import pytest


def sample():
    rows = []
    def add(source, identity, score=60, date='2026-06-08', tokens=100, hash_value=None, partition='2026/24'):
        rows.append(dict(source=source, doc_id=identity, aliveness_score=score,
                         created_dt=date, feat_token_count=tokens,
                         content_hash=hash_value, crawl_partition=partition))
    add('reddit', 'a', 10, hash_value='same')
    add('reddit', 'b', 90, hash_value='same')
    add('reddit', 'c', 50, tokens=50)
    add('reddit', 'short', tokens=49)
    for i in range(4):
        add('fourchan', str(i), i * 20, hash_value='broken')
    add('fourchan', '0', 0, hash_value='broken')
    add('common_crawl', 'cc', date=None)
    add('news', 'missing', date=None)
    add('news', 'old', date='1999-01-01')
    add('news', 'future', date='2027-01-01')
    for src in ['wikipedia', 'wayback', 'github']:
        add(src, src)
    return pd.DataFrame(rows)


def test_filters_dedup_fallback_and_exact_quantiles():
    from pipeline.timeline_aggregates import aggregate_timeline
    monthly, weekly, meta = aggregate_timeline(sample(), now='2026-10-17', min_n_crawl=1, min_n_biweekly=1)
    month = monthly.set_index('source')
    week = weekly.set_index('source')
    assert set(month.index) == {'common_crawl'}
    assert week.loc['reddit', 'n'] == 2
    assert week.loc['reddit', 'median'] == 30
    assert week.loc['reddit', 'p25'] == 20
    assert week.loc['reddit', 'p75'] == 40
    assert week.loc['fourchan', 'n'] == 4
    assert 'common_crawl' not in week.index
    assert month.loc['common_crawl', 'crawl_id'] == '2026-24'
    assert meta['input_row_count'] == len(sample())
    assert meta['dedup']['duplicates_dropped'] == 2
    assert meta['rows_dropped']['null_date'] == 1
    assert meta['rows_dropped']['pre_2000'] == 1
    assert meta['rows_dropped']['future_date'] == 1
    assert meta['rows_dropped']['excluded_source'] == 3


def test_default_minimum_bin_counts():
    from pipeline.timeline_aggregates import aggregate_timeline
    monthly, weekly, _ = aggregate_timeline(sample(), now='2026-10-17')
    assert monthly.empty and weekly.empty


def test_parquet_build_reads_only_projected_columns_in_batches(tmp_path, monkeypatch):
    from pipeline import timeline_aggregates as timeline
    import pyarrow.parquet as pq
    df = sample()
    df['text'] = 'must never be loaded'
    path = tmp_path / 'scored.parquet'
    df.to_parquet(path)
    original = pq.ParquetFile.iter_batches
    calls = []
    def projected(self, *args, **kwargs):
        calls.append(kwargs['columns'])
        assert 'text' not in kwargs['columns']
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', projected)
    timeline.build_timeline(path, tmp_path / 'out', now='2026-10-17')
    assert calls
    assert (tmp_path / 'out' / 'meta.json').exists()
    assert pd.read_csv(tmp_path / 'out' / 'biweekly.csv').empty


def test_empty_fourchan_hash_is_reported_as_degenerate():
    from pipeline.timeline_aggregates import aggregate_timeline
    df = sample().loc[lambda x: x.source.eq('fourchan')].copy()
    df['content_hash'] = ''
    _, _, meta = aggregate_timeline(df, now='2026-10-17')
    assert meta['dedup']['fourchan_degenerate_hashes'] == ['']
    assert meta['dedup']['fourchan_fallback_rows'] == 5


def test_hash_majority_is_global_and_dedup_crosses_batch_boundaries(tmp_path):
    from pipeline.timeline_aggregates import build_timeline
    df = pd.concat([sample().loc[lambda x: x.source.eq('fourchan')],
                    sample().loc[lambda x: x.source.eq('reddit')]], ignore_index=True)
    path = tmp_path / 'scored.parquet'
    df.to_parquet(path)
    monthly, weekly, meta = build_timeline(path, tmp_path / 'out', batch_size=2,
                                           now='2026-10-17', min_n_biweekly=1, min_n_crawl=1)
    assert weekly.set_index('source').loc['fourchan', 'n'] == 4
    assert weekly.set_index('source').loc['reddit', 'n'] == 2
    assert meta['dedup']['duplicates_dropped'] == 2
    assert meta['dedup']['fourchan_fallback_rows'] == 5


def test_half_shared_fourchan_hash_is_not_degenerate():
    from pipeline.timeline_aggregates import aggregate_timeline
    df = sample().loc[lambda x: x.source.eq('fourchan')].iloc[:4].copy()
    df['content_hash'] = ['same', 'same', 'other', 'last']
    _, weekly, meta = aggregate_timeline(df, now='2026-10-17', min_n_biweekly=1)
    assert weekly.iloc[0]['n'] == 3
    assert meta['dedup']['fourchan_degenerate_hashes'] == []


def test_iso_crawl_weeks_validate_week_53_and_year_boundary():
    from pipeline.timeline_aggregates import aggregate_timeline
    df = pd.concat([sample().loc[lambda x: x.source.eq('common_crawl')]] * 3, ignore_index=True)
    df['doc_id'] = ['a', 'b', 'c']
    df['crawl_partition'] = ['2020/53', '2021/53', '2026/01']
    monthly, _, meta = aggregate_timeline(df, now='2026-10-17', min_n_crawl=1)
    assert monthly['median_date'].tolist() == ['2020-12-28', '2025-12-29']
    assert meta['rows_dropped']['null_date'] == 1


def test_sample_size_threshold_boundaries():
    from pipeline.timeline_aggregates import aggregate_timeline
    df = pd.concat([sample().iloc[:1]] * 100, ignore_index=True)
    df['doc_id'] = df.index.astype(str)
    df['content_hash'] = df.index.astype(str)
    assert aggregate_timeline(df.iloc[:99], now='2026-10-17')[1].empty
    assert aggregate_timeline(df, now='2026-10-17')[1].iloc[0]['n'] == 100
    df['source'] = 'common_crawl'
    assert aggregate_timeline(df.iloc[:49], now='2026-10-17')[0].empty
    assert aggregate_timeline(df.iloc[:50], now='2026-10-17')[0].iloc[0]['n'] == 50


def test_bootstrap_deterministic_and_alignment():
    from pipeline.timeline_aggregates import bootstrap_ci, aggregate_timeline
    assert bootstrap_ci(range(101)) == bootstrap_ci(range(101))
    lo, hi = bootstrap_ci(range(101))
    assert lo <= 50 <= hi
    df = pd.concat([sample().iloc[:1]] * 4, ignore_index=True)
    df['content_hash'] = df.index.astype(str)
    df['created_dt'] = ['2026-01-05', '2026-01-18', '2026-01-19', '2026-01-04']
    _, bins, _ = aggregate_timeline(df, now='2026-10-17', min_n_biweekly=1)
    assert bins.set_index('period_start')['n'].to_dict() == {'2025-12-22': 1, '2026-01-05': 2, '2026-01-19': 1}

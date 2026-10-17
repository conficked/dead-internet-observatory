"""Build small per-source timelines without reading document text.

Deduplication keeps the first occurrence in file order, before date/token filters.
Two passes allow detection of globally degenerate 4chan hashes across batches.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ['doc_id', 'source', 'created_dt', 'crawl_partition', 'aliveness_score',
           'feat_token_count', 'content_hash']
SOURCES = {'common_crawl', 'bluesky', 'fourchan', 'hackernews', 'mastodon',
           'reddit', 'steam', 'youtube', 'stackoverflow', 'news'}


def bootstrap_ci(scores, seed=0, resamples=1000):
    values = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    if len(values) > 20000:
        values = rng.choice(values, 20000, replace=False)
    # Bound temporary memory while preserving exactly 1000 resamples.
    medians = np.concatenate([np.median(rng.choice(values, (min(50, resamples-i), len(values))), axis=1)
                              for i in range(0, resamples, 50)])
    return tuple(np.quantile(medians, [.025, .975]))


def _summarise(bins, date_column, min_n, crawl_dates=None):
    records = []
    for (source, date), scores in sorted(bins.items()):
        if len(scores) < min_n:
            continue
        p25, median, p75 = np.quantile(scores, [.25, .5, .75])
        low, high = bootstrap_ci(scores)
        row = dict(source=source, n=len(scores), median=median, ci_low=low, ci_high=high, p25=p25, p75=p75)
        if crawl_dates is not None:
            row.update(crawl_id=date, median_date=pd.Timestamp(int(np.median(crawl_dates[date])), tz='UTC').strftime('%Y-%m-%d'))
        else:
            row[date_column] = date
        records.append(row)
    columns = (['source', 'crawl_id', 'median_date'] if crawl_dates is not None else ['source', date_column])
    return pd.DataFrame(records, columns=columns + ['n', 'median', 'ci_low', 'ci_high', 'p25', 'p75'])


def _aggregate(batches, *, now=None, min_tokens=50, min_tokens_crawl=100,
               min_n_crawl=50, min_n_biweekly=100):
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    hashes = Counter()
    fourchan_count = 0
    for frame in batches(['source', 'content_hash']):
        values = frame.loc[frame['source'].eq('fourchan'), 'content_hash']
        fourchan_count += len(values)
        hashes.update(values.dropna())
    degenerate = sorted(h for h, n in hashes.items() if n > fourchan_count / 2)
    fallback_fourchan = bool(degenerate)
    seen = set()
    monthly, weekly = defaultdict(list), defaultdict(list)
    crawl_dates = defaultdict(list)
    dropped = Counter({key: 0 for key in ['excluded_source', 'duplicate', 'null_date',
                      'pre_2000', 'future_date', 'invalid_score', 'below_min_tokens', 'missing_doc_id']})
    dedup = Counter(duplicates_dropped=0, doc_id_fallback_rows=0, fourchan_fallback_rows=0)
    input_count = 0
    month_eligible = week_eligible = 0
    fallback_dates = 0
    for frame in batches(COLUMNS):
        input_count += len(frame)
        allowed = frame['source'].isin(SOURCES)
        dropped['excluded_source'] += int((~allowed).sum())
        frame = frame.loc[allowed].copy()
        keep = []
        for row in frame[['source', 'content_hash', 'doc_id']].itertuples(index=False):
            source, fingerprint, doc_id = row
            fallback = pd.isna(fingerprint) or fingerprint == '' or (source == 'fourchan' and fallback_fourchan)
            if fallback:
                dedup['doc_id_fallback_rows'] += 1
                dedup['fourchan_fallback_rows'] += int(source == 'fourchan' and fallback_fourchan)
                if pd.isna(doc_id) or doc_id == '':
                    dropped['missing_doc_id'] += 1
                    keep.append(False)
                    continue
            key = (source, 'id' if fallback else 'hash', str(doc_id if fallback else fingerprint))
            duplicate = key in seen
            if duplicate:
                dedup['duplicates_dropped'] += 1
                dropped['duplicate'] += 1
            else:
                seen.add(key)
            keep.append(not duplicate)
        frame = frame.loc[keep].copy()
        dates = pd.to_datetime(frame['created_dt'], errors='coerce', utc=True, format='mixed')
        missing_cc = dates.isna() & frame['source'].eq('common_crawl')
        # Partitions here are ISO year/week, never month numbers or ingestion dates.
        partitions = frame.loc[missing_cc, 'crawl_partition'].astype('string')
        valid_week = partitions.str.fullmatch(r'\d{4}/(?:0[1-9]|[1-4]\d|5[0-3])').fillna(False)
        fallback = pd.to_datetime(partitions.where(valid_week) + '/1', format='%G/%V/%u', errors='coerce', utc=True)
        # Reject invalid ISO week 53 in years which only have 52 weeks.
        roundtrip = fallback.dt.strftime('%G/%V').eq(partitions)
        fallback = fallback.where(roundtrip)
        fallback_dates += int(fallback.notna().sum())
        dates.loc[missing_cc] = fallback
        for reason, mask in [('null_date', dates.isna()),
                             ('pre_2000', dates < pd.Timestamp('2000-01-01', tz='UTC')),
                             ('future_date', dates > now)]:
            dropped[reason] += int(mask.sum())
        valid = dates.notna() & dates.ge(pd.Timestamp('2000-01-01', tz='UTC')) & dates.le(now)
        frame = frame.loc[valid].copy()
        dates = dates.loc[valid]
        scores = pd.to_numeric(frame['aliveness_score'], errors='coerce')
        finite = np.isfinite(scores)
        dropped['invalid_score'] += int((~finite).sum())
        frame = frame.loc[finite].copy()
        scores, dates = scores.loc[finite], dates.loc[finite]
        tokens = pd.to_numeric(frame['feat_token_count'], errors='coerce')
        dropped['below_min_tokens'] += int((~tokens.ge(min(min_tokens, min_tokens_crawl))).sum())
        anchor = pd.Timestamp('2026-01-05', tz='UTC')
        periods = (anchor + pd.to_timedelta(((dates - anchor).dt.days // 14) * 14, unit='D')).dt.strftime('%Y-%m-%d')
        for source, partition, date, period, score, length in zip(
                frame['source'], frame['crawl_partition'], dates, periods, scores, tokens):
            if source == 'common_crawl' and length >= min_tokens_crawl and pd.notna(partition):
                cid = str(partition).replace('/', '-')
                monthly[(source, cid)].append(float(score))
                crawl_dates[cid].append(date.value)
                month_eligible += 1
            if source != 'common_crawl' and length >= min_tokens and date.year == 2026:
                weekly[(source, period)].append(float(score))
                week_eligible += 1
    meta = {
        'build_time': datetime.now(timezone.utc).isoformat(),
        'date_cutoff_utc': now.isoformat(), 'input_row_count': input_count,
        'rows_dropped': dict(dropped),
        'dedup': {**dict(dedup), 'fourchan_degenerate_hashes': degenerate,
                  'policy': 'first occurrence per source/hash; doc_id for missing hashes or degenerate fourchan source'},
        'common_crawl_partition_fallback_rows': fallback_dates,
        'thresholds': {'min_tokens': min_tokens, 'min_tokens_crawl': min_tokens_crawl,
                       'min_n_crawl': min_n_crawl, 'min_n_biweekly': min_n_biweekly,
                       'biweekly_year': 2026, 'bin_days': 14, 'alignment_anchor': '2026-01-05', 'bootstrap_resamples': 1000, 'bootstrap_seed': 0, 'bootstrap_max_sample': 20000, 'earliest_date': '2000-01-01'},
        'eligible_rows': {'cc_crawls': month_eligible, 'biweekly': week_eligible},
        'bins_dropped_below_min_n': {
            'cc_crawls': sum(len(v) < min_n_crawl for v in monthly.values()),
            'biweekly': sum(len(v) < min_n_biweekly for v in weekly.values())},
        'date_rules': 'Common Crawl WARC created_dt, ISO crawl week fallback; social created_dt; future dates excluded',
        'excluded_sources': ['wikipedia', 'wayback', 'github'],
    }
    return _summarise(monthly, 'median_date', min_n_crawl, crawl_dates), _summarise(weekly, 'period_start', min_n_biweekly), meta


def aggregate_timeline(df, **kwargs):
    """Aggregate a small in-memory frame using the same rules as the CLI."""
    return _aggregate(lambda columns: iter([df[columns]]), **kwargs)


def build_timeline(input_path, out, *, batch_size=65536, **kwargs):
    parquet = pq.ParquetFile(input_path)
    def batches(columns):
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()
    monthly, weekly, meta = _aggregate(batches, **kwargs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out / 'cc_crawls.csv', index=False)
    weekly.to_csv(out / 'biweekly.csv', index=False)
    for obsolete in ('monthly.csv', 'weekly.csv'):
        (out / obsolete).unlink(missing_ok=True)
    (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return monthly, weekly, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', default='data/timeline/')
    parser.add_argument('--min-tokens', type=int, default=50)
    parser.add_argument('--min-tokens-crawl', type=int, default=100)
    args = parser.parse_args()
    monthly, weekly, meta = build_timeline(args.input, args.out, min_tokens=args.min_tokens,
                                          min_tokens_crawl=args.min_tokens_crawl)
    print(f"Read {meta['input_row_count']:,} rows; wrote {len(monthly)} crawl and {len(weekly)} biweekly bins to {args.out}")


if __name__ == '__main__':
    main()

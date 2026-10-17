# Dead Internet Observatory

**Observational signals in sampled public web text**

The [Dead Internet Theory](https://en.wikipedia.org/wiki/Dead_Internet_theory) holds that the internet is increasingly populated by bots, AI-generated content, and automated engagement farms rather than real humans.

This project computes an experimental **Internet Aliveness Index (IAI)** from linguistic signals in collected public web text. The 0–100 score is observational, not a calibrated AI probability or an estimate of the fraction of the internet that is synthetic. Coverage varies by source and collection date.

No paid model APIs or subscriptions. Detection runs locally with classical NLP
signals plus optional DistilGPT-2 perplexity scoring.

---

## The Dashboard

> Live at: **[deadinternetobservatory.streamlit.app](https://deadinternetobservatory.streamlit.app/)**

- Current published score and individually dated source means from a SQLite snapshot
- Common Crawl per-crawl medians and bootstrap confidence intervals for the observed 2014–2026 record
- Per-platform two-week medians for 2026, plus a fixed equal-weight panel composite only when every panel source meets the minimum sample count
- Methodology, coverage and limitations

---

## How the Index Works

The IAI uses seven classical linguistic and behavioural signals:

| Signal | Weight | Feature interpretation |
|---|---|---|
| **Type-Token Ratio** | 18% | Vocabulary diversity |
| **Shannon Entropy** | 15% | Character distribution diversity |
| **Sentence Length Variance** | 15% | Variation in sentence lengths |
| **Bigram Repetition** | 15% | Repeated adjacent word pairs |
| **Temporal Burstiness** | 15% | Timing variation (neutral when timestamps are absent) |
| **MTLD** | 12% | Lexical diversity; length effects can remain |
| **Zipf Law Alignment** | 10% | Fit to a ranked word-frequency distribution |

All signals are statistical. No external model calls, no GPU required. Everything runs on standard CPU in Python.

When enabled with `ENABLE_PERPLEXITY=1`, local DistilGPT-2 perplexity receives
15% of the score and scales the classical weights proportionally. Common Crawl
disables perplexity to limit runtime. Scoring modes and genre affect comparability.
Short texts score lower; token thresholds reduce but do not remove length confounding.

---

## Local timeline aggregates

```bash
python -m pipeline.timeline_aggregates --input /path/to/scored.parquet --out data/timeline/
```

The builder projects seven metadata/score columns in PyArrow batches and never
reads `text`. The small `cc_crawls.csv`, `biweekly.csv` and `meta.json` outputs are
tracked in this repository and loaded locally by the dashboard. Rebuild them
when the scored input changes; the current workflow does not commit them.

Timelines keep the first occurrence per `(source, content_hash)`, falling back
to `doc_id` for missing hashes or a degenerate 4chan hash shared by more than half
of that source's rows. Common Crawl uses `created_dt` (WARC-Date), with an ISO
`YYYY/WW` crawl-week fallback when absent; social sources use `created_dt`.
Null, pre-2000 and future dates are dropped. Wikipedia, Wayback and GitHub are
excluded. Each Common Crawl point groups one `crawl_partition` and is plotted at
its median WARC date, requiring ≥100 tokens per document and n≥50. Lines and
bands break only when consecutive crawl points are more than 120 days apart.
The ChatGPT marker is 2022-11-30; the caption computes period summaries from the
data and describes a pre-existing gradual decline only when those summaries support it.

The main band is a bootstrap 95% confidence interval of the median: 1000 resamples,
fixed seed 0, with a deterministic subsample of at most 20000 documents per bin.
The optional faint p25–p75 band describes document spread, distinct from uncertainty
in the median. Neither interval calibrates AI authorship probabilities.

The 2026 platform series excludes Common Crawl and uses 14-day bins aligned to
Monday 2026-01-05, requiring ≥50 tokens and n≥100. A single selected platform
shows its bootstrap CI. The composite is the equal-weight mean of Hacker News,
Mastodon, Steam, YouTube, Bluesky and 4chan medians, shown only for bins when all
six have n≥100. Missing panel sources are never reweighted. Reddit may be selected
individually but is outside this fixed panel. Axes fit the displayed values with padding.
Token thresholds can be changed with `--min-tokens` and `--min-tokens-crawl`.

Common Crawl has an independent 50000-document pipeline quota and a 10000-row
per-crawl cap counting existing gold rows, selected round-robin across crawls.
Other sources retain the newest-first 6000-document quota. CC remains perplexity-free.
The minion harvests up to five crawls per run, five WET files per crawl and 2000
records per file, stopping at the remaining per-crawl quota. Failed files receive
three retries with backoff, then replacement files. Attempts and successful segment
counts are persisted in `config/cc_attempts.json`; a 12-hour cooldown allows the
pipeline to publish confirmation. A crawl becomes done only when published gold
rows reach 80% of target (a smaller target requires every available WET file to be
exhausted without truncation or errors). Published counts live in SQLite table
`cc_crawl_status` within `observatory.db`, uploaded by the existing publisher.
The minion reads that public database from `conficked/dead-internet-observatory`
with retry/backoff; fetch failures never confirm completion. Pipeline CI restores
and publishes to the same dataset. No uploads are needed to rebuild local timelines.
The scoped backfill removes exactly the 65 diagnosed done-but-empty IDs, leaving
other done entries intact. Progress does not automatically reset:
`sources.common_crawl.auto_reset_progress: false`; manual workflow dispatch offers
`reset_progress` (default false). A completed or cooling-down queue exits cleanly.

SQLite delta ingestion tracks document IDs and recomputes exact daily statistics.
For an older aggregate-only database, deltas merge into existing daily summaries
using document-weighted means, percentages and approximate medians, plus pooled
variance, with a warning reporting the number of legacy groups merged. Delta
document IDs are tracked to prevent re-ingestion from counting them twice; rebuild
with `ingest_scored_df` when full historical documents are available for exact statistics.

---

## Architecture

```
+---------------------------------------------------------------------------+
|                             DATA COLLECTION MINIONS                      |
|  Common Crawl · Reddit · News RSS · Wikipedia · HackerNews · Wayback      |
|  Bluesky · 4chan · Steam · YouTube · LinkedIn · Stack Overflow             |
|  Mastodon · GitHub Content                                                 |
+-----------------------------------+---------------------------------------+
                                    | raw JSONL -> GitHub Artifacts
                                    v
+---------------------------------------------------------------------------+
|                               PIPELINE                                    |
|  Bronze (raw JSONL) -> Silver (normalised Parquet)                        |
|                     -> Gold (scored Parquet, 6000 docs/run cap)           |
|                     -> SQLite index snapshot                              |
+-------------+-----------------------------+-----------------------------+
              | Parquet + SQLite -> Hugging Face Datasets
              v                             v
+--------------------+         +-------------------------------------+
|  HuggingFace       |         |  Streamlit Community Cloud          |
|  Datasets          |         |  (reads published SQLite snapshot)  |
|  (public, free)    |         |                                     |
+--------------------+         +-------------------------------------+
```

GitHub Actions runs the full cycle autonomously. No server required.

```
01:00 UTC daily   ->  Wayback Machine
Every 4 hours     ->  Common Crawl (up to 5 segments/date)
03:00 UTC daily   ->  Reddit
03:30 UTC daily   ->  News Crawler
04:00 UTC daily   ->  Wikipedia + HackerNews
05:00 UTC daily   ->  Daily Full Sweep (fast minions)
6x daily          ->  Pipeline (bronze->silver->gold->Hugging Face)
06:00 UTC daily   ->  Stack Overflow
06:30 UTC daily   ->  Bluesky
07:00 UTC daily   ->  4chan
07:00/13:00/21:00 ->  Mastodon
07:30 UTC daily   ->  Steam
08:00/14:00/20:00 ->  YouTube
08:30 UTC daily   ->  LinkedIn
10:00 UTC daily   ->  GitHub Content
18:00 UTC daily   ->  Stack Overflow (repeat)
22:00 UTC daily   ->  GitHub Content (repeat)
```

---

## Data Sources

| Minion | Source | What we collect | Status |
|---|---|---|---|
| `common_crawl_bot` | [Common Crawl](https://commoncrawl.org) | Quarterly WET snapshots, raw web text, multi-year backfill | ✓ |
| `reddit_bot` | Reddit public JSON API | Posts + comment trees from 10 subreddits | ✓ |
| `news_crawler_bot` | 8 RSS feeds | Full article text from major news outlets | ✓ |
| `wikipedia_bot` | Wikipedia API | 500 random articles + 24h edit pattern tracking | ✓ |
| `hackernews_bot` | [Algolia HN API](https://hn.algolia.com/api) | Stories + comments, last 7 days | ✓ |
| `wayback_bot` | [Wayback Machine CDX API](https://web.archive.org/cdx) | 14 sentinel URLs × 5 years, longitudinal signal | ✓ |
| `bluesky_bot` | Bluesky public API | Posts across 5 topic search terms | ✓ |
| `fourchan_bot` | 4chan JSON API | Posts from 8 boards | ✓ |
| `steam_bot` | Steam Reviews API | Reviews across 16 popular games | ✓ |
| `youtube_bot` | YouTube Data API v3 | Comments across 5 topic searches, 3× daily | ✓ |
| `linkedin_bot` | LinkedIn public feeds | Article text | ✓ |
| `stackoverflow_bot` | StackExchange API v2.3 | Q&A across 10 tags | ✓ |
| `mastodon_bot` | 5 Mastodon instances | Public timelines + 5 tag streams | ✓ |
| `github_bot` | GitHub REST API | README + top issues from trending repos | ✓ |
| `twitter_bot` | Twitter/X | — | ✗ Cloudflare-blocked |
| `substack_bot` | Substack | — | ✗ Cloudflare-blocked on GH Actions IPs |

**Where data lives:**
- Raw JSONL (bronze) → GitHub Artifacts, normally 7-day retention; Common Crawl uses 1 day
- Processed Parquet (silver/gold) → [HuggingFace Datasets](https://huggingface.co/datasets/jupiternull/dead-internet-observatory)
- Daily index aggregates → Hugging Face `observatory.db`
- Scored doc registry → Hugging Face `gold/doc_registry.parquet`

---

## Data Privacy and Security

The collectors only access publicly available pages and public APIs. Published
Silver and Gold Parquet files contain full collected text and may include stable
public account identifiers, URLs, titles, domains, timestamps, and personal or
sensitive information embedded in public content. Collection does not sanitize
or redact these fields before publication.

- Secrets are supplied through GitHub Actions secrets or environment variables.
- Credentials are never required by the public Streamlit dashboard.
- `.env`, key, certificate, and Streamlit secret files are excluded from git.
- GitHub secret scanning, push protection, private vulnerability reporting, and
  Dependabot vulnerability alerts are enabled for this repository.
- Forks must use their own credentials and must not publish them in configuration files.

To request removal of a dataset record, open a GitHub issue containing only its
`doc_id`. Do not post sensitive text or personal details. Credential exposures
and other security issues should use the repository's private vulnerability
reporting form under the **Security** tab. Removal from prior Hugging Face
commits may require a separate history rewrite.

---

## Quick Start

```bash
git clone https://github.com/jupiternull/dead-internet-observatory
cd dead-internet-observatory
pip install -r requirements-streamlit.txt
streamlit run app/app.py
```

The dashboard reads the published SQLite snapshot from Hugging Face. To run data collection locally, see below.

---

## Running Real Data Collection

```bash
# Individual minions
python3 -m minions.reddit_bot
python3 -m minions.news_crawler_bot
python3 -m minions.wikipedia_bot
python3 -m minions.hackernews_bot
python3 -m minions.bluesky_bot
python3 -m minions.fourchan_bot
python3 -m minions.steam_bot
python3 -m minions.youtube_bot
python3 -m minions.linkedin_bot
python3 -m minions.stackoverflow_bot
python3 -m minions.mastodon_bot
python3 -m minions.github_bot
python3 -m minions.wayback_bot        # slow — fetches 5 years of snapshots
python3 -m minions.common_crawl_bot   # heavy — downloads WET files

# Run the full pipeline after harvesting
python3 -m pipeline.bronze_ingestion
python3 -m pipeline.silver_processing

# Publish to your Hugging Face dataset repository
HF_TOKEN=... python3 scripts/push_to_hf.py --repo owner/dataset
```

---

## Deployment

**Streamlit Community Cloud (free):**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to `app/app.py`
3. Deploy — Streamlit reads the public Hugging Face dataset snapshot and needs no secrets

By default, a fork displays the upstream `jupiternull/dead-internet-observatory`
dataset. For an independent deployment, create a Hugging Face dataset repository
and replace the dataset ID in `app/app.py`, `scripts/push_to_hf.py`, and
`.github/workflows/pipeline_and_index.yml`.

**Activating autonomous data collection (GitHub Actions):**
1. Enable Actions and scheduled workflows on the fork
2. Pre-create the Hugging Face dataset repository
3. Add a write-capable `HF_TOKEN` for dataset publication
4. Add `YOUTUBE_API_KEY` to enable YouTube collection
5. Optionally add `STACKOVERFLOW_API_KEY` for higher Stack Exchange quotas

`GITHUB_TOKEN` is supplied automatically by GitHub Actions. Supabase and
`DATABASE_URL` are no longer used.

Never commit these values. Add them under repository **Settings → Secrets and
variables → Actions**.

---

Dependency files are split by runtime:
- `requirements-streamlit.txt` — dashboard deployment
- `requirements-minions.txt` — source collectors
- `requirements-pipeline.txt` — processing, scoring, and Hugging Face publication
- `requirements.txt` — full superset kept for existing deployment compatibility

---

## Repo Structure

```
dead-internet-observatory/
├── minions/           # data collection bots
├── pipeline/          # Bronze → Silver → Gold ETL + SQLite index
├── detection/         # Statistical AI-content detection engine
├── analytics/         # Aliveness Index computation + anomaly detection
├── app/               # Streamlit dashboard
├── scripts/           # Hugging Face publication and migration helpers
├── config/            # config.yaml — all tuneable parameters
└── .github/workflows/ # Autonomous GitHub Actions workflows
```

---

## Contributing

This is a living project. A few things that would move the needle:

- **Better signal weights** — calibrate against labeled human/AI datasets (HC3, RAID, M4 on HF)
- **New minions** — Internet Archive bulk data, academic paper feeds, forum scrapers
- **Perplexity calibration** — benchmark DistilGPT-2 scoring against labeled corpora
- **Anomaly alerts** — webhook or email when the IAI drops more than X points in a day

Open an issue or send a PR.

---

## License

Source code is MIT licensed. Redistributed web content remains subject to its
original source terms and applicable law; the MIT license does not grant rights
to third-party dataset content.

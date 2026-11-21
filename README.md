# Dead Internet Observatory

**How the web actually dies.**

A link-rot census. A fixed sample of web pages captured by Common Crawl between 2013 and
2026 is revisited every month to see which pages still exist, which have changed, and which
are gone, and whether the Wayback Machine kept a copy of the ones that died.

**Live site: [conficked.github.io/dead-internet-observatory](https://conficked.github.io/dead-internet-observatory/)**

"Dead internet" usually means a theory about bots. This project measures something you can
check: pages that existed, and now don't.

---

## What it measures

| Question | How |
|---|---|
| How much of the web from year X is still online? | Fate of the sample, grouped by capture year |
| What is the half-life of a web page? | Exponential fit of survival against page age |
| How long does a page keep its original wording? | The same fit, counting only pages whose text still matches |
| How do pages die? | 404s, soft 404s, redirects to the home page, parked domains, dead domains |
| Did anyone keep a copy? | Wayback Machine availability for a random subset of dead pages |

Every page gets one of fourteen fates in four groups:

| Group | Fates |
|---|---|
| **alive** | intact, edited, moved, reachable |
| **drifted** | replaced (same address, different page) |
| **dead** | not found, soft 404, sent home, parked, defunct, domain gone |
| **unknown** | blocked, server error, unreachable (left out of every share) |

Text comparison measures how much of the original wording is still on the page. The sample
keeps a bottom-32 sketch of each page's three-word shingles from the text Common Crawl
captured; the checker counts how many of those 32 hashes appear in the live page's text.
At least 0.75 is intact, at least 0.35 is edited, less is replaced. This holds up when the
same article gets a new site design around it. A front page that still answers counts as
alive whatever it says, since front pages are meant to change. The thresholds are heuristics
and are published on the site.

## How it runs

```
gold/scored.parquet (Hugging Face, 4.3M docs)
        │  linkrot.sample: up to 1,000 URLs per crawl, chosen by salted URL hash
        ▼
linkrot/sample.jsonl.gz ──► 8 parallel shards: linkrot.check
                              one polite GET per URL, classify the fate,
                              Wayback lookups for a random subset of the dead
        ▼
linkrot.aggregate ──► linkrot/site/*.json on Hugging Face ──► GitHub Pages
                  └─► linkrot/census/census-DATE.jsonl.gz on Hugging Face
```

Everything runs in GitHub Actions (`.github/workflows/linkrot.yml`), monthly and on demand.
No servers, no paid APIs. The checker uses only the Python standard library.

Polite by design: an identifying user agent, at most two requests in flight per site, a
15 second timeout, a 512 KB cap on every response, and one visit per page per month.

### Run it locally

```bash
pip install -r requirements-linkrot.txt
python -m linkrot.hf get linkrot/sample.jsonl.gz work/sample.jsonl.gz
python -m linkrot.check --sample work/sample.jsonl.gz --out work/results-0.jsonl --limit 200
python -m linkrot.aggregate --sample work/sample.jsonl.gz --results 'work/results-*.jsonl' --site site/data
python -m http.server -d site 8000
```

Tests: `pytest tests/test_linkrot.py` (includes an end-to-end run against a local server
that serves every fate). `python -m linkrot.report` prints a tuning report; the census
workflow runs it, so the job log shows examples of each fate.

### Site

A static page in `site/`: black, monospace, built from characters. Glyph density carries the
data (`█` alive, `▒` changed, `░` dead but archived, `·` dead and lost). The departures board
is a custom [ascii.rest](https://ascii.rest) piece (`site/pieces/departed.js`, adapted from its
split-flap board, MIT); TV static and the candle are ascii.rest library pieces.

### Data

- Sample: `linkrot/sample.jsonl.gz` in the [Hugging Face dataset](https://huggingface.co/datasets/conficked/dead-internet-observatory)
- Each census: `linkrot/census/census-YYYY-MM-DD.jsonl.gz`, one row per URL with its fate,
  HTTP status, redirect target, text distance and archive status
- Site data: `linkrot/site/summary.json`, `graveyard.json`, `history.json` (pulled into
  `site/data/` at deploy time; `master` is protected, so data never goes through git)

## Limits

- The sample covers what Common Crawl captured and what this project pulled from each crawl
  (English pages only). It is not a random sample of the whole web.
- Every visit comes from one cloud network. Sites that block data-center traffic show up as
  unknown, not dead.
- Soft 404 and parking detection use text patterns and miss cases in both directions.
- "Replaced" can mean a genuinely new page or the same article inside a new site design.

## History

This repository began as an experimental "Internet Aliveness Index" that scored text for
signs of machine generation. That work found the score tracked genre and site mix more than
authorship. Its write-up lives in [docs/aliveness-index.md](docs/aliveness-index.md); the
collectors and pipeline that built the 4.3M-document corpus still run and feed this census.

To request removal of a record, open an issue containing only its URL hash (`id`).

## License

Code is MIT licensed. Collected web content remains subject to its original terms.

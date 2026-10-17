"""
Common Crawl Minion — downloads and processes WET (plain-text) files.

Common Crawl releases quarterly snapshots of the entire public web.
WET files are pre-extracted plaintext — no HTML parsing needed.
We sample a configurable number of WET segments per crawl date,
stream each file, parse WARC records, filter to English, and save
to the bronze layer as JSONL.

Free quota: CC data is 100% free via S3 (no egress from us-east-1)
or via HTTPS. We use HTTPS for simplicity.
"""

import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import gzip
import io
import random
import re
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse

import requests

from .base_minion import BaseMinion


class CommonCrawlMinion(BaseMinion):

    CC_BASE = "https://data.commoncrawl.org"
    ENGLISH_STOPWORDS = frozenset({
        "the", "is", "are", "was", "were", "and", "or", "but", "in", "on",
        "at", "to", "for", "of", "a", "an", "that", "this", "with", "from",
        "by", "as", "it", "its", "not", "be", "have", "had", "has", "do",
        "did", "will", "would", "can", "could", "should", "may", "might",
    })

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path, "common_crawl")
        cfg = self.config["sources"]["common_crawl"]
        self.crawl_dates: List[str] = cfg["crawl_dates"]
        self.max_segments: int = cfg["max_segments_per_date"]
        self.wet_per_segment: int = cfg.get("wet_files_per_segment", 5)
        self.max_per_wet: int = cfg.get("max_records_per_wet", 2000)
        self.min_text: int = self.config["detection"]["min_text_length"]

    # ── WET path discovery ────────────────────────────────────────────────────

    def get_wet_paths(self, crawl_id: str) -> List[str]:
        """Fetch the master WET paths index for a crawl snapshot."""
        url = f"{self.CC_BASE}/crawl-data/CC-MAIN-{crawl_id}/wet.paths.gz"
        self.logger.info(f"Fetching WET index for CC-MAIN-{crawl_id} …")
        try:
            resp = requests.get(url, timeout=90, stream=True)
            if resp.status_code == 404:
                self.logger.warning(f"  WET index unavailable for CC-MAIN-{crawl_id}")
                return []
            resp.raise_for_status()
            content = gzip.decompress(resp.content).decode("utf-8")
            paths = [ln.strip() for ln in content.splitlines() if ln.strip()]
            self.logger.info(f"  Found {len(paths):,} WET segments")
            return paths
        except Exception as exc:
            self.logger.error(f"  Could not fetch WET index: {exc}")
            return []

    # ── WET streaming ─────────────────────────────────────────────────────────

    def stream_wet_records(self, wet_path: str) -> Iterator[Dict]:
        """Stream and yield parsed records from a single gzipped WET file."""
        url = f"{self.CC_BASE}/{wet_path}"
        self.logger.info(f"  Streaming {wet_path.split('/')[-1]} …")

        try:
            resp = requests.get(url, timeout=180, stream=True)
            resp.raise_for_status()

            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                buf.write(chunk)
            buf.seek(0)

            with gzip.GzipFile(fileobj=buf) as gz:
                raw = gz.read().decode("utf-8", errors="replace")

            count = 0
            for block in raw.split("WARC/1.0\r\n"):
                if count >= self.max_per_wet:
                    break
                record = self._parse_wet_block(block)
                if record:
                    yield record
                    count += 1

            self.logger.info(f"    → Parsed {count:,} records from segment")

        except Exception as exc:
            self.logger.error(f"  Error streaming {wet_path}: {exc}")
            self.stats["errors"] += 1
            raise

    def _parse_wet_block(self, block: str) -> Optional[Dict]:
        """Parse one WARC/WET block into a structured record."""
        if "WARC-Type: conversion" not in block:
            return None

        # Split headers from body at first blank line
        header_section, _, body = block.partition("\r\n\r\n")
        headers: Dict[str, str] = {}
        for line in header_section.splitlines():
            if ": " in line:
                k, _, v = line.partition(": ")
                headers[k.strip()] = v.strip()

        url = headers.get("WARC-Target-URI", "")
        date = headers.get("WARC-Date", "")
        body = body.strip()

        if not url or len(body) < self.min_text:
            return None

        domain = self._extract_domain(url)
        if not domain or not self._is_likely_english(body):
            return None

        # Classify domain category
        category = self._categorize_domain(domain, url)

        return {
            "url": url,
            "domain": domain,
            "category": category,
            "date": date,
            "text": body[: self.config["detection"]["max_text_length_for_features"]],
            "text_length": len(body),
            "content_hash": self.content_hash(body),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        try:
            return urlparse(url).netloc.lower().lstrip("www.")
        except Exception:
            return None

    def _is_likely_english(self, text: str) -> bool:
        sample_words = set(text[:600].lower().split())
        return len(sample_words & self.ENGLISH_STOPWORDS) >= 4

    @staticmethod
    def _categorize_domain(domain: str, url: str) -> str:
        news_tlds = {"reuters.com", "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com",
                     "theguardian.com", "washingtonpost.com", "apnews.com", "npr.org"}
        social = {"reddit.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
                  "tiktok.com", "linkedin.com", "tumblr.com", "mastodon.social"}
        wiki = {"wikipedia.org", "wikimedia.org"}
        blog = {"medium.com", "substack.com", "wordpress.com", "blogspot.com"}

        if any(domain.endswith(n) for n in news_tlds):
            return "news"
        if any(domain.endswith(s) for s in social):
            return "social"
        if any(domain.endswith(w) for w in wiki):
            return "wiki"
        if any(domain.endswith(b) for b in blog):
            return "blog"
        return "web"

    # ── Entry point ───────────────────────────────────────────────────────────

    PROGRESS_FILE = "config/cc_progress.json"

    def _all_known_crawl_ids(self) -> List[str]:
        """Configured historical dates + any newer CC releases not yet in config."""
        known = []
        configured = list(self.crawl_dates)
        try:
            resp = requests.get(
                "https://index.commoncrawl.org/collinfo.json", timeout=30
            )
            resp.raise_for_status()
            available = set()
            for entry in resp.json():
                cid = entry["id"].replace("CC-MAIN-", "")
                if not re.fullmatch(r"\d{4}-\d{2}", cid):
                    continue
                available.add(cid)
            invalid = [cid for cid in configured if cid not in available]
            if invalid:
                self.logger.warning(f"Skipping unavailable Common Crawl IDs: {invalid}")
            known = [cid for cid in configured if cid in available]
            for cid in sorted(available, reverse=True):
                if cid not in known:
                    known.append(cid)
        except Exception as exc:
            self.logger.warning(f"Could not fetch collinfo: {exc}")
            known = configured
        return known

    def _load_progress(self) -> set:
        import json
        from pathlib import Path
        p = Path(self.PROGRESS_FILE)
        if p.exists():
            return set(json.loads(p.read_text()))
        return set()

    def _save_progress(self, done: set):
        import json
        from pathlib import Path
        Path(self.PROGRESS_FILE).write_text(json.dumps(sorted(done), indent=2))

    ATTEMPTS_FILE = "config/cc_attempts.json"
    STATUS_URL = "https://huggingface.co/datasets/conficked/dead-internet-observatory/resolve/main/observatory.db"

    def _published_counts(self):
        for attempt in range(3):
            try:
                response = requests.get(self.STATUS_URL, timeout=90)
                response.raise_for_status()
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'status.db'
                    path.write_bytes(response.content)
                    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
                        return dict(conn.execute('SELECT crawl_id, gold_rows FROM cc_crawl_status'))
            except Exception as exc:
                self.logger.warning(f'Published CC status unavailable: {exc}')
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    def run(self, dry_run: bool = False):
        cfg = self.config['sources']['common_crawl']
        if not cfg.get('enabled', True):
            return
        all_ids = self._all_known_crawl_ids()
        done = self._load_progress()
        attempt_path = Path(self.ATTEMPTS_FILE)
        attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
        if os.environ.get('CC_RESET_PROGRESS', '').lower() == 'true':
            done, attempts = set(), {}
        counts = self._published_counts()
        target = cfg.get('max_rows_per_crawl', 10000)
        if counts is not None:
            for cid, rows in counts.items():
                available = attempts.get(cid, {}).get('available_records', target)
                if rows >= .8 * min(target, available) and min(target, available) > 0:
                    done.add(cid)
        remaining = [cid for cid in all_ids if cid not in done]
        if not remaining and cfg.get('auto_reset_progress', False):
            done, attempts = set(), {}
            remaining = list(all_ids)
        if not dry_run:
            self._save_progress(done)
        now = datetime.now(timezone.utc)
        remaining = [cid for cid in remaining if cid not in attempts or
                     (now - datetime.fromisoformat(attempts[cid]['last_attempt_iso'])).total_seconds() >= 12*3600]
        if not remaining:
            self.logger.info('[CC] Nothing remains to harvest; awaiting published gold confirmation or new crawls.')
            return
        for cid in remaining[:cfg.get('max_dates_per_run', 5)]:
            paths = self.get_wet_paths(cid)
            if not paths:
                attempts[cid] = dict(last_attempt_iso=now.isoformat(), segments_ok=0, records=0)
                if not dry_run:
                    attempt_path.write_text(json.dumps(attempts, indent=2) + "\n")
                continue
            records, segments_ok, exhausted_records = 0, 0, 0
            all_untruncated = True
            need = max(0, target - (counts or {}).get(cid, 0))
            # Shuffle all paths so failed segments can be replaced without reuse.
            for path in random.sample(paths, len(paths)):
                if segments_ok >= self.max_segments or records >= need:
                    break
                batch = None
                for retry in range(4):  # initial request plus three retries
                    try:
                        batch = list(self.stream_wet_records(path))
                        break
                    except Exception as exc:
                        self.logger.warning(f'{cid}: segment failed ({retry+1}/4): {exc}')
                        if retry < 3:
                            time.sleep(2 ** retry)
                if batch is None:
                    continue
                segments_ok += 1
                all_untruncated = all_untruncated and len(batch) < self.max_per_wet
                exhausted_records += len(batch)
                batch = batch[:need-records]
                records += len(batch)
                self.stats['fetched'] += len(batch)
                if batch and not dry_run:
                    self.save_bronze(batch, 'common_crawl', cid.replace('-', '/'))
                    self.stats['processed'] += len(batch)
                self.throttle(3.0)
            attempt = dict(last_attempt_iso=now.isoformat(), segments_ok=segments_ok, records=records)
            # A smaller target is valid only after exhausting every available file,
            # with no failures or per-file truncation.
            if segments_ok == len(paths) and all_untruncated:
                attempt['available_records'] = exhausted_records
            attempts[cid] = attempt
            if not dry_run:
                attempt_path.write_text(json.dumps(attempts, indent=2) + '\n')
        self.report_stats()


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    CommonCrawlMinion().run(dry_run=dry)

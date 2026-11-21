"""Draw a deterministic, per-crawl sample of Common Crawl URLs from gold/scored.parquet.

Each crawl contributes at most ``per_crawl`` URLs: those with the smallest salted
hash of the normalized URL. The choice depends only on the URL and the salt, so a
rerun picks the same pages, and adding new crawls never reshuffles old ones. A URL
seen in several crawls is kept once, in its earliest crawl.

    python -m linkrot.sample --input gold/scored.parquet --out linkrot/sample.jsonl.gz
"""

import argparse
import gzip
import hashlib
import heapq
import json
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional
from urllib.parse import urlsplit, urlunsplit

from linkrot.fingerprint import simhash, sketch, to_hex

COLUMNS = ["source", "url", "domain", "category", "crawl_partition", "created_dt",
           "text", "text_length"]
DEFAULT_SALT = "linkrot-v1"


def normalize_url(url: str) -> Optional[str]:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    netloc = host if parts.port in (None, 80, 443) else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def rank_key(url: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}|{url}".encode("utf-8")).digest()[:8], "big")


def host_of(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


def tld_of(host: str) -> str:
    return host.rsplit(".", 1)[-1] if "." in host else host


def crawl_of(row: Dict) -> Optional[str]:
    crawl = row.get("crawl_partition")
    if not crawl:
        return None
    return str(crawl).replace("/", "-")


def iso_date(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).date().isoformat()
    text = str(value)
    return text[:10] if len(text) >= 10 else None


def last_words(text: str, limit: int = 90) -> str:
    """A short first line of the page, used only to label a grave."""
    line = " ".join((text or "").strip().splitlines()[0].split()) if text and text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1].rsplit(" ", 1)[0] + "…"


def slim(row: Dict) -> Dict:
    return {k: row.get(k) for k in ("text", "text_length", "created_dt", "category")}


def select(rows: Iterable[Dict], per_crawl: int, salt: str = DEFAULT_SALT,
           existing: Optional[List[Dict]] = None) -> List[Dict]:
    """Keep the ``per_crawl`` lowest-ranked Common Crawl URLs in each crawl.

    With ``existing``, the selection is additive: every existing record is kept as it is,
    URLs already in it are skipped, and each crawl only fills the room it has left.
    """
    existing = existing or []
    tracked = {r["url"] for r in existing}
    held: Dict[str, int] = {}
    for r in existing:
        held[r["crawl"]] = held.get(r["crawl"], 0) + 1
    heaps: Dict[str, list] = {}
    members: Dict[str, set] = {}
    for row in rows:
        if row.get("source") != "common_crawl":
            continue
        url = normalize_url(row.get("url") or "")
        crawl = crawl_of(row)
        if not url or not crawl:
            continue
        room = per_crawl - held.get(crawl, 0)
        if url in tracked or room <= 0:
            continue
        key = rank_key(url, salt)
        heap = heaps.setdefault(crawl, [])
        seen = members.setdefault(crawl, set())
        if url in seen:
            continue
        # Keep only the fields a record needs; fingerprints are computed once, after selection.
        if len(heap) < room:
            heapq.heappush(heap, (-key, url, slim(row)))
            seen.add(url)
        elif -heap[0][0] > key:
            _, dropped, _ = heapq.heapreplace(heap, (-key, url, slim(row)))
            seen.discard(dropped)
            seen.add(url)

    chosen: Dict[str, tuple] = {}
    for crawl in sorted(heaps):
        for _, url, row in heaps[crawl]:
            if url in chosen and chosen[url][0] <= crawl:
                continue
            chosen[url] = (crawl, row)
    added = [record(url, crawl, row) for url, (crawl, row) in chosen.items()]
    return sorted(existing + added, key=lambda r: (r["crawl"], r["id"]))


def record(url: str, crawl: str, row: Dict) -> Dict:
    text = row.get("text") or ""
    host = host_of(url)
    return {
        "id": url_id(url),
        "url": url,
        "host": host,
        "tld": tld_of(host),
        "crawl": crawl,
        "crawl_date": iso_date(row.get("created_dt")),
        "category": row.get("category") or "web",
        "text_length": int(row.get("text_length") or len(text)),
        "simhash": to_hex(simhash(text)),
        "sketch": sketch(text),
        "last_words": last_words(text),
    }


def iter_parquet(path: str, batch_size: int = 4096) -> Iterator[Dict]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as parquet:
        names = set(parquet.schema_arrow.names)
        columns = [c for c in COLUMNS if c in names]
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            mask = pc.equal(batch.column(columns.index("source")), "common_crawl")
            batch = batch.filter(mask)
            if batch.num_rows:
                yield from batch.to_pylist()


def write_jsonl_gz(path: str, rows: Iterable[Dict]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl_gz(path: str) -> Iterator[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="gold/scored.parquet")
    parser.add_argument("--out", required=True, help="output .jsonl.gz")
    parser.add_argument("--per-crawl", type=int, default=1000)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--existing", default="", help="sample to keep and extend (additive refresh)")
    args = parser.parse_args(argv)

    existing = list(read_jsonl_gz(args.existing)) if args.existing else []
    rows = select(iter_parquet(args.input), args.per_crawl, args.salt, existing)
    count = write_jsonl_gz(args.out, rows)
    crawls = len({r["crawl"] for r in rows})
    old_crawls = len({r["crawl"] for r in existing})
    print(f"[sample] {count:,} URLs from {crawls} crawls -> {args.out} "
          f"(kept {len(existing):,}, added {count - len(existing):,}; {crawls - old_crawls} new crawls)")


if __name__ == "__main__":
    main()

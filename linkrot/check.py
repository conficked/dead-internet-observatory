"""Check a shard of the sample: fetch each URL once, classify its fate, and append results.

Polite by design: one GET per URL, an identifying User-Agent, at most two requests
in flight per host, a byte cap on every body, and a resumable output file.

    python -m linkrot.check --sample sample.jsonl.gz --out results-0.jsonl --shard 0 --shards 8
"""

import argparse
import http.client
import json
import os
import random
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from linkrot import USER_AGENT
from linkrot.classify import Fetch, classify
from linkrot.sample import read_jsonl_gz

MAX_BYTES = 512 * 1024
TIMEOUT = 15
DEADLINE = 40
MAX_REDIRECTS = 10
PER_HOST = 2


_LOCAL = threading.local()


class _CountingRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            _LOCAL.redirects = getattr(_LOCAL, "redirects", 0) + 1
        return new


def _opener(verify: bool):
    context = ssl.create_default_context()
    if not verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(
        _CountingRedirects(), urllib.request.HTTPSHandler(context=context))


_OPENERS = {True: _opener(True), False: _opener(False)}


def _error_kind(exc: BaseException) -> str:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, socket.gaierror):
        # EAI_NONAME / EAI_NODATA mean the name does not resolve; others are transient.
        return "dns_nxdomain" if reason.errno in (socket.EAI_NONAME, getattr(socket, "EAI_NODATA", -5)) else "dns_temp"
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(reason, ConnectionRefusedError):
        return "refused"
    if isinstance(reason, ssl.SSLError):
        return "tls"
    if isinstance(reason, (ConnectionResetError, http.client.RemoteDisconnected, BrokenPipeError)):
        return "reset"
    if isinstance(reason, str) and "timed out" in reason:
        return "timeout"
    return "other"


def _read(response, started: float) -> bytes:
    chunks, total = [], 0
    while total < MAX_BYTES:
        if time.monotonic() - started > DEADLINE:
            break
        chunk = response.read(min(65536, MAX_BYTES - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _decode(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    for part in content_type.split(";"):
        part = part.strip().lower()
        if part.startswith("charset="):
            charset = part.split("=", 1)[1].strip("\"' ") or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def fetch(url: str, verify: bool = True) -> Fetch:
    started = time.monotonic()
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        "Accept-Language": "en;q=0.9,*;q=0.5",
    })
    result = Fetch()
    _LOCAL.redirects = 0
    try:
        with _OPENERS[verify].open(request, timeout=TIMEOUT) as response:
            result.http_status = response.status
            result.final_url = response.geturl()
            result.content_type = response.headers.get("Content-Type", "").lower()
            raw = _read(response, started)
    except urllib.error.HTTPError as exc:
        result.http_status = exc.code
        result.final_url = exc.geturl() or url
        result.content_type = (exc.headers.get("Content-Type", "") if exc.headers else "").lower()
        try:
            raw = _read(exc, started)
        except Exception:
            raw = b""
    except Exception as exc:
        kind = _error_kind(exc)
        if kind == "tls" and verify:
            retry = fetch(url, verify=False)
            retry.tls_invalid = True
            return retry
        result.error = kind
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        return result

    result.redirects = getattr(_LOCAL, "redirects", 0)
    if "html" in result.content_type or "text" in result.content_type or not result.content_type:
        result.body = _decode(raw, result.content_type)
    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    return result


def wayback_closest(url: str, date: Optional[str], attempts: int = 3) -> Dict:
    """Ask the Wayback Machine for its closest capture to the crawl date."""
    params = {"url": url}
    if date:
        params["timestamp"] = date.replace("-", "")
    endpoint = "https://archive.org/wayback/available?" + urllib.parse.urlencode(params)
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read(200_000).decode("utf-8", errors="replace"))
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}
            if closest.get("available"):
                return {"archived": True, "archive_ts": closest.get("timestamp")}
            return {"archived": False}
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt + 1 < attempts:
                time.sleep(20 * (attempt + 1))
                continue
            return {"archived": None}
        except Exception:
            if attempt + 1 < attempts:
                time.sleep(5 * (attempt + 1))
                continue
            return {"archived": None}
    return {"archived": None}


class HostGate:
    """At most ``limit`` requests in flight per host."""

    def __init__(self, limit: int = PER_HOST):
        self.limit = limit
        self.lock = threading.Lock()
        self.gates: Dict[str, threading.Semaphore] = {}

    def __call__(self, host: str) -> threading.Semaphore:
        with self.lock:
            return self.gates.setdefault(host, threading.Semaphore(self.limit))


def check_one(rec: Dict, gate: HostGate) -> Dict:
    with gate(rec.get("host", "")):
        got = fetch(rec["url"])
    v = classify(rec["url"], rec.get("simhash"), got, rec.get("sketch"), rec.get("last_words", ""))
    out = {
        "id": rec["id"],
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": v.status,
        "bucket": v.bucket,
        "http_status": got.http_status,
        "error": got.error,
        "redirects": got.redirects,
        "distance": v.distance,
        "retained": v.retained,
        "tls_invalid": got.tls_invalid,
        "elapsed_ms": got.elapsed_ms,
        "title": v.title[:160],
    }
    if got.final_url and got.final_url != rec["url"]:
        out["final_url"] = got.final_url[:500]
    if v.notes:
        out["notes"] = v.notes
    return out


def in_shard(rec_id: str, shard: int, shards: int) -> bool:
    return int(rec_id, 16) % shards == shard


def read_jsonl(path: str) -> List[Dict]:
    """Read a plain JSONL file, stopping quietly at a torn last line."""
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(records: Iterable[Dict], out_path: str, workers: int = 48,
        time_budget: Optional[float] = None, seed: int = 0) -> Dict:
    """Check every record not already in ``out_path`` (plain JSONL, appended line by line)."""
    records = list(records)
    already = {row["id"] for row in read_jsonl(out_path)}
    if already:  # drop a torn tail before appending
        kept = read_jsonl(out_path)
        with open(out_path, "w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in kept)
    todo = [r for r in records if r["id"] not in already]
    random.Random(seed).shuffle(todo)  # spread hosts and crawls over time
    gate = HostGate()
    started = time.monotonic()
    counts: Dict[str, int] = {}
    stop = threading.Event()

    def task(rec):
        if stop.is_set():
            return None
        return check_one(rec, gate)

    with open(out_path, "a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, rec): rec for rec in todo}
        for n, future in enumerate(as_completed(futures), 1):
            rec = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # never lose a shard to one URL
                row = {"id": rec["id"], "status": "unreachable", "bucket": "unknown",
                       "error": f"crash:{type(exc).__name__}", "checked_at": _now()}
            if row is None:
                continue
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")
            sink.flush()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if n % 500 == 0:
                rate = n / max(1e-9, time.monotonic() - started)
                print(f"[check] {n:,}/{len(todo):,} ({rate:.1f}/s) {dict(sorted(counts.items()))}", flush=True)
            if time_budget and not stop.is_set() and time.monotonic() - started > time_budget:
                print("[check] time budget reached; finishing in-flight requests (a rerun resumes)", flush=True)
                stop.set()
    return {"checked": sum(counts.values()), "previously_done": len(already), "counts": counts,
            "seconds": round(time.monotonic() - started, 1)}


def run_wayback(records: Dict[str, Dict], results: List[Dict], out_path: str,
                cap: int = 600, rate: float = 0.5, seed: int = 0) -> int:
    """Look up a capped random subset of dead or drifted pages in the Wayback Machine."""
    candidates = [r for r in results if r.get("bucket") in ("dead", "drifted") and r["id"] in records]
    random.Random(seed + 1).shuffle(candidates)
    done = 0
    with open(out_path, "w", encoding="utf-8") as sink:
        for row in candidates[:cap]:
            rec = records[row["id"]]
            found = wayback_closest(rec["url"], rec.get("crawl_date"))
            sink.write(json.dumps({"id": row["id"], **found}, separators=(",", ":")) + "\n")
            sink.flush()
            done += 1
            if done % 100 == 0:
                print(f"[wayback] {done}/{min(len(candidates), cap)}", flush=True)
            if rate:
                time.sleep(1.0 / rate)
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True, help="results .jsonl (appended, resumable)")
    parser.add_argument("--wayback-out", default="", help="wayback .jsonl (optional)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0, help="check at most N URLs (testing)")
    parser.add_argument("--wayback-cap", type=int, default=600)
    parser.add_argument("--wayback-rate", type=float, default=0.5, help="lookups per second")
    parser.add_argument("--time-budget", type=float, default=0, help="seconds before stopping")
    args = parser.parse_args(argv)

    records = [r for r in read_jsonl_gz(args.sample) if in_shard(r["id"], args.shard, args.shards)]
    if args.limit:  # a spread across crawls, not the oldest N
        records = sorted(records, key=lambda r: r["id"])[: args.limit]
    print(f"[check] shard {args.shard}/{args.shards}: {len(records):,} URLs", flush=True)
    summary = run(records, args.out, workers=args.workers,
                  time_budget=args.time_budget or None, seed=args.shard)
    print(f"[check] done: {json.dumps(summary)}", flush=True)
    if args.wayback_out and args.wayback_cap:
        n = run_wayback({r["id"]: r for r in records}, read_jsonl(args.out), args.wayback_out,
                        cap=args.wayback_cap, rate=args.wayback_rate, seed=args.shard)
        print(f"[wayback] done: {n} lookups", flush=True)


if __name__ == "__main__":
    main()

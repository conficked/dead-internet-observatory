"""Merge the sample with check results and write the site's data files.

    python -m linkrot.aggregate --sample sample.jsonl.gz --results 'results/*.jsonl' \
        --wayback 'results/wayback-*.jsonl' --site site/data --ledger census.jsonl.gz
"""

import argparse
import glob
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional

from linkrot.classify import (BUCKETS, EDITED_MAX, EDITED_MIN_RETAINED, INTACT_MAX,
                              INTACT_MIN_RETAINED)
from linkrot.sample import read_jsonl_gz, write_jsonl_gz

BUCKET_ORDER = ["alive", "drifted", "dead", "unknown"]
STATUS_ORDER = ["intact", "edited", "moved", "reachable", "replaced", "redirect_home",
                "soft_404", "not_found", "parked", "defunct", "domain_gone",
                "blocked", "server_error", "unreachable"]
UNSAFE = re.compile(
    r"(porn|xxx|sex|nude|naked|escort|adult|cam(girl|s)\b|milf|hentai|fetish|nsfw|"
    r"casino|gambl|poker|betting|viagra|cialis|pharma|pills|replica|payday|loan)", re.I)


def read_jsonl(paths: Iterable[str]) -> List[Dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> List[Optional[float]]:
    if not n:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def share(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def parse_day(value: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None


def fit_half_life(points: List[Dict], key: str) -> Optional[Dict]:
    """Fit S(t) = exp(-lambda * t) through the origin, weighted by sample size."""
    num = den = 0.0
    for p in points:
        s, t, n = p.get(key), p["age_years"], p["known"]
        if s is None or not 0 < s < 1 or t <= 0 or n < 30:
            continue
        num += n * t * -math.log(s)
        den += n * t * t
    if den <= 0 or num <= 0:
        return None
    lam = num / den
    return {"lambda_per_year": round(lam, 5), "half_life_years": round(math.log(2) / lam, 2)}


def merge(sample: List[Dict], results: List[Dict], wayback: List[Dict]) -> List[Dict]:
    by_id = {r["id"]: dict(r) for r in sample}
    for row in results:  # later rows win, so a recheck replaces an earlier attempt
        if row["id"] in by_id:
            by_id[row["id"]].update({k: v for k, v in row.items() if k != "id"})
    for row in wayback:
        if row["id"] in by_id and row.get("archived") is not None:
            by_id[row["id"]]["archived"] = row["archived"]
            by_id[row["id"]]["archive_ts"] = row.get("archive_ts")
    return list(by_id.values())


def summarize(rows: List[Dict], census_day: date) -> Dict:
    checked = [r for r in rows if r.get("status") in BUCKETS]
    status_counts = Counter(r["status"] for r in checked)
    bucket_counts = Counter(r["bucket"] for r in checked)
    known = sum(bucket_counts[b] for b in ("alive", "drifted", "dead"))

    years = defaultdict(list)
    for r in checked:
        day = parse_day(r.get("crawl_date"))
        if day:
            years[day.year].append((r, day))

    by_year = []
    for year in sorted(years):
        items = years[year]
        b = Counter(r["bucket"] for r, _ in items)
        s = Counter(r["status"] for r, _ in items)
        k = b["alive"] + b["drifted"] + b["dead"]
        ages = [(census_day - d).days / 365.25 for _, d in items]
        unchanged = s["intact"] + s["edited"] + s["moved"]
        by_year.append({
            "year": year, "n": len(items), "known": k,
            "age_years": round(sum(ages) / len(ages), 2),
            "buckets": {name: b[name] for name in BUCKET_ORDER},
            "statuses": {name: s[name] for name in STATUS_ORDER if s[name]},
            "dead_share": share(b["dead"], k), "dead_ci": wilson(b["dead"], k),
            "alive_share": share(b["alive"], k),
            "survival": share(b["alive"] + b["drifted"], k),
            "unchanged": share(unchanged, k),
            "unknown_share": share(b["unknown"], len(items)),
        })

    tld = defaultdict(Counter)
    for r in checked:
        tld[r.get("tld") or "?"][r["bucket"]] += 1
    by_tld = []
    for name, c in tld.items():
        k = c["alive"] + c["drifted"] + c["dead"]
        if k >= 150:
            by_tld.append({"tld": name, "known": k, "dead_share": share(c["dead"], k),
                           "dead_ci": wilson(c["dead"], k)})
    by_tld.sort(key=lambda x: -x["known"])
    by_tld = sorted(by_tld[:14], key=lambda x: -(x["dead_share"] or 0))

    cat = defaultdict(Counter)
    for r in checked:
        cat[r.get("category") or "web"][r["bucket"]] += 1
    by_category = []
    for name, c in sorted(cat.items()):
        k = c["alive"] + c["drifted"] + c["dead"]
        if k >= 50:
            by_category.append({"category": name, "known": k, "dead_share": share(c["dead"], k)})

    dead_total = bucket_counts["dead"]
    dead_breakdown = {s: share(status_counts[s], dead_total)
                      for s in STATUS_ORDER if BUCKETS[s] == "dead" and status_counts[s]}

    looked = [r for r in checked if r.get("archived") is not None]
    dead_looked = [r for r in looked if r["bucket"] == "dead"]
    archived_dead = sum(1 for r in dead_looked if r["archived"])
    archive_share = share(archived_dead, len(dead_looked))
    dead_share = share(dead_total, known)
    archive = {
        "dead_checked": len(dead_looked), "dead_archived": archived_dead,
        "dead_archived_share": archive_share, "dead_archived_ci": wilson(archived_dead, len(dead_looked)),
        "lost_share_of_known": round(dead_share * (1 - archive_share), 4)
        if dead_share is not None and archive_share is not None else None,
    }

    hist = Counter()
    for r in checked:
        if r.get("distance") is not None:
            hist[min(r["distance"], 40) // 2 * 2] += 1
    distance_hist = [{"from": k, "n": hist[k]} for k in range(0, 42, 2)]

    kept = Counter()
    for r in checked:
        if r.get("retained") is not None:
            kept[min(int(r["retained"] * 10), 9)] += 1
    retained_hist = [{"from": k / 10, "n": kept[k]} for k in range(10)]

    hosts = len({r.get("host") for r in checked})
    crawl_days = sorted(d for d in (parse_day(r.get("crawl_date")) for r in rows) if d)
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "census_date": census_day.isoformat(),
        "sample": {
            "urls": len(rows), "checked": len(checked), "hosts": hosts,
            "crawls": len({r.get("crawl") for r in rows}),
            "first_crawl_date": crawl_days[0].isoformat() if crawl_days else None,
            "last_crawl_date": crawl_days[-1].isoformat() if crawl_days else None,
        },
        "totals": {
            "statuses": {s: status_counts[s] for s in STATUS_ORDER if status_counts[s]},
            "buckets": {b: bucket_counts[b] for b in BUCKET_ORDER},
            "known": known,
        },
        "headline": {
            "dead_share": dead_share, "dead_ci": wilson(dead_total, known),
            "alive_share": share(bucket_counts["alive"], known),
            "drifted_share": share(bucket_counts["drifted"], known),
            "intact_share": share(status_counts["intact"], known),
            "unknown_share": share(bucket_counts["unknown"], len(checked)),
            "domain_gone_share": share(status_counts["domain_gone"], known),
        },
        "decay": {
            "survival": fit_half_life(by_year, "survival"),
            "unchanged": fit_half_life(by_year, "unchanged"),
        },
        "by_year": by_year,
        "by_tld": by_tld,
        "by_category": by_category,
        "dead_breakdown": dead_breakdown,
        "archive": archive,
        "distance_hist": distance_hist,
        "retained_hist": retained_hist,
        "method": {"intact_min_retained": INTACT_MIN_RETAINED, "edited_min_retained": EDITED_MIN_RETAINED,
                   "intact_max_distance": INTACT_MAX, "edited_max_distance": EDITED_MAX,
                   "sketch_size": 32, "shingle_words": 3},
    }


def graveyard(rows: List[Dict], limit: int = 360, seed: int = 0) -> List[Dict]:
    """A stratified, filtered set of dead pages for the site's graveyard."""
    dead = [r for r in rows if r.get("bucket") == "dead"
            and not UNSAFE.search(f"{r.get('url', '')} {r.get('last_words', '')}")]
    years = defaultdict(list)
    for r in dead:
        day = parse_day(r.get("crawl_date"))
        if day:
            years[day.year].append(r)
    rng = random.Random(seed)
    per_year = max(1, limit // max(1, len(years)))
    chosen = []
    for year in sorted(years):
        items = sorted(years[year], key=lambda r: r["id"])
        rng.shuffle(items)
        chosen.extend(items[:per_year])
    out = []
    for r in chosen:
        stamp = (r.get("crawl_date") or "").replace("-", "")
        out.append({
            "url": r["url"], "host": r.get("host"), "crawl_date": r.get("crawl_date"),
            "status": r["status"], "last_words": r.get("last_words") or "",
            "archived": r.get("archived"),
            "wayback": f"https://web.archive.org/web/{stamp}/{r['url']}",
        })
    return out


def update_history(path: str, summary: Dict) -> List[Dict]:
    history = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            history = json.load(stream)
    entry = {"census_date": summary["census_date"], "checked": summary["sample"]["checked"],
             **{k: summary["headline"][k] for k in ("dead_share", "alive_share", "drifted_share",
                                                     "unknown_share")}}
    history = [h for h in history if h["census_date"] != entry["census_date"]] + [entry]
    return sorted(history, key=lambda h: h["census_date"])


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", required=True)
    parser.add_argument("--results", required=True, help="glob of results .jsonl files")
    parser.add_argument("--wayback", default="", help="glob of wayback .jsonl files")
    parser.add_argument("--site", default="site/data")
    parser.add_argument("--ledger", default="", help="optional merged .jsonl.gz output")
    args = parser.parse_args(argv)

    sample = list(read_jsonl_gz(args.sample))
    results = read_jsonl(sorted(glob.glob(args.results)))
    wayback = read_jsonl(sorted(glob.glob(args.wayback))) if args.wayback else []
    rows = merge(sample, results, wayback)
    days = [parse_day(r.get("checked_at")) for r in results]
    days = [d for d in days if d]
    census_day = max(days) if days else datetime.now(timezone.utc).date()

    summary = summarize(rows, census_day)
    write_json(os.path.join(args.site, "summary.json"), summary)
    write_json(os.path.join(args.site, "graveyard.json"), graveyard(rows))
    history_path = os.path.join(args.site, "history.json")
    write_json(history_path, update_history(history_path, summary))
    if args.ledger:
        write_jsonl_gz(args.ledger, rows)
    h = summary["headline"]
    print(f"[aggregate] {summary['sample']['checked']:,} checked; dead {h['dead_share']}, "
          f"alive {h['alive_share']}, drifted {h['drifted_share']}, unknown {h['unknown_share']}")


if __name__ == "__main__":
    main()

"""Print a diagnostic report of a census to the job log, for tuning the classifier.

    python -m linkrot.report --sample sample.jsonl.gz --results 'results-*.jsonl'
"""

import argparse
import glob
import random
from collections import Counter

from linkrot.aggregate import merge, read_jsonl
from linkrot.sample import read_jsonl_gz


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args(argv)

    rows = [r for r in merge(list(read_jsonl_gz(args.sample)), read_jsonl(sorted(glob.glob(args.results))), [])
            if r.get("status")]
    rng = random.Random(0)
    print(f"\n==== census report: {len(rows):,} checked ====")
    print("status:", dict(Counter(r["status"] for r in rows).most_common()))
    print("unknown causes:", dict(Counter(
        r.get("error") or f"http_{r.get('http_status')}" for r in rows if r["bucket"] == "unknown").most_common(15)))
    print("notes:", dict(Counter(n for r in rows for n in r.get("notes", [])).most_common()))
    kept = Counter(min(int(r["retained"] * 10), 9) for r in rows if r.get("retained") is not None)
    print("retained deciles:", {f"{k / 10:.1f}": kept[k] for k in range(10)})
    print("tls_invalid:", sum(1 for r in rows if r.get("tls_invalid")))
    for status in ("replaced", "edited", "reachable", "soft_404", "redirect_home", "parked", "defunct", "blocked"):
        pick = [r for r in rows if r["status"] == status]
        rng.shuffle(pick)
        if not pick:
            continue
        print(f"\n-- {status} ({len(pick)}) --")
        for r in pick[: args.examples]:
            print(f"  kept={r.get('retained')} d={r.get('distance')} http={r.get('http_status')} "
                  f"{r['url'][:90]} -> {(r.get('final_url') or '')[:60]} | {r.get('title', '')[:60]!r} "
                  f"| was {r.get('last_words', '')[:50]!r}")


if __name__ == "__main__":
    main()

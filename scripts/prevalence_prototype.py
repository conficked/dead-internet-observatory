"""Prototype: estimate the AI-generated share of Common Crawl text, with placebo checks.

    .venv/bin/python scripts/prevalence_prototype.py sample
    OPENROUTER_API_KEY=... .venv/bin/python scripts/prevalence_prototype.py generate
    .venv/bin/python scripts/prevalence_prototype.py evaluate --output data/prevalence/report.json

Eras by crawl date for evaluate (--reference-years, default 2019-2021):
  reference         human occurrence probabilities; a held-out part calibrates the
                    estimator's bias
  placebo_forward   pre-ChatGPT crawls after the reference years: must estimate ~0
  placebo_backward  crawls before the reference years: measures drift backwards
  target            crawls from 2023: the measurement
Generation seeds keep their original split: AI reference seeds from 2013-2019 pages,
AI test seeds from 2020 to 2022 pages.

AI reference text comes from REFERENCE_MODELS. The injection test also uses
UNSEEN_MODELS, which the occurrence model never sees.
"""

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.analysis_io import write_json
from analytics.prevalence import (
    ConfidenceBelt, OccurrenceModel, dictionary_vocabulary, document_window, estimate_alpha,
    injection_test, main_text, placebo_verdict, replicate_intervals, trend_placebo,
)

DATA = ROOT / "data" / "prevalence"
SAMPLE = DATA / "cc_sample.parquet"
SPREAD_SAMPLE = DATA / "cc_spread_sample.parquet"
DATASET = "datasets/conficked/dead-internet-observatory/gold/scored.parquet"
# Common-Crawl-only row groups of gold/scored.parquet, one or two per era.
ROW_GROUPS = (345, 350, 390, 385, 369, 402, 356, 359, 365, 379, 374, 395)
COLUMNS = ["doc_id", "domain", "url", "text", "crawl_partition", "aliveness_score"]

REFERENCE_MODELS = ("anthropic/claude-haiku-5.5", "openai/gpt-6-luna",
                    "deepseek/deepseek-v4.1-flash")
UNSEEN_MODELS = ("google/gemma-4-31b-it", "qwen/qwen3.8-flash")
CHATGPT_RELEASE = (2022, 48)

PROMPT = """Below is the text of a web page. Write the full text of a different web page \
of the same kind: same type of site, same topic and audience, about {words} words. \
Write only the page text as it would appear on the site, in plain paragraphs. \
No preamble, no commentary, no markdown. Do not copy sentences from the original.

---
{seed}
---"""


def crawl_key(partition):
    year, week = str(partition).split("/")[:2]
    return int(year), int(week)


def split_of(partition):
    key = crawl_key(partition)
    if key[0] <= 2019:
        return "reference"
    if key < CHATGPT_RELEASE:
        return "placebo"
    if key[0] >= 2023:
        return "target"
    return "transition"


def load_sample():
    frame = pd.read_parquet(SAMPLE)
    frame = frame.drop_duplicates("doc_id")
    frame["split"] = frame["crawl_partition"].map(split_of)
    return frame


def cmd_sample(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    DATA.mkdir(parents=True, exist_ok=True)
    with HfFileSystem().open(DATASET, "rb", block_size=2 ** 22) as handle:
        parquet = pq.ParquetFile(handle)
        tables = []
        for group in ROW_GROUPS:
            table = parquet.read_row_group(group, columns=COLUMNS + ["source"])
            tables.append(table.filter(pa.compute.equal(table["source"], "common_crawl"))
                          .select(COLUMNS))
            print(f"row group {group}: {tables[-1].num_rows:,} rows")
    pq.write_table(pa.concat_tables(tables), SAMPLE)
    print(f"wrote {SAMPLE}")


def scan_row_groups(path=DATA / "rg_coverage.parquet"):
    """Common Crawl rows per (row group, crawl). Reads only two narrow columns."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    with HfFileSystem().open(DATASET, "rb", block_size=2 ** 22) as handle:
        parquet = pq.ParquetFile(handle)
        for group in range(parquet.metadata.num_row_groups):
            meta = parquet.metadata.row_group(group)
            table = parquet.read_row_group(group, columns=["source", "crawl_partition"]).to_pandas()
            counts = table[table["source"].eq("common_crawl")].groupby("crawl_partition").size()
            rows.extend((group, crawl, int(n), meta.total_byte_size / meta.num_rows)
                        for crawl, n in counts.items())
    frame = pd.DataFrame(rows, columns=["rg", "crawl", "n", "bytes_per_row"])
    frame.to_parquet(path)
    return frame


def choose_row_groups(coverage, crawls, min_groups, min_docs, cap_per_group):
    """Greedy cover: every crawl drawn from at least `min_groups` row groups (or all it
    has) and about `min_docs` rows, preferring groups dense in still-needed crawls.
    Row groups are pipeline batches, so spreading a crawl across them lets the
    bootstrap see between-batch composition differences."""
    coverage = coverage[coverage["crawl"].isin(crawls)]
    need_groups = {c: min(min_groups, g["rg"].nunique()) for c, g in coverage.groupby("crawl")}
    need_docs = {c: min(min_docs, int(g["n"].sum())) for c, g in coverage.groupby("crawl")}
    got_groups = dict.fromkeys(need_groups, 0)
    got_docs = dict.fromkeys(need_groups, 0)
    groups = {rg: g for rg, g in coverage.groupby("rg")}
    cost = {rg: float((g["n"] * g["bytes_per_row"]).sum()) / 3e7 for rg, g in groups.items()}
    chosen = []
    while True:
        best, best_score = None, 0.3
        for rg, g in groups.items():
            if rg in chosen:
                continue
            gain = sum((got_groups[c] < need_groups[c])
                       + min(n, cap_per_group, max(0, need_docs[c] - got_docs[c])) / 1000
                       for c, n in zip(g["crawl"], g["n"]))
            if gain / cost[rg] > best_score:
                best, best_score = rg, gain / cost[rg]
        if best is None:
            return chosen
        chosen.append(best)
        for c, n in zip(groups[best]["crawl"], groups[best]["n"]):
            got_groups[c] += 1
            got_docs[c] += min(n, cap_per_group)


def cmd_sample_spread(args):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    crawls = set(pd.read_parquet(SAMPLE, columns=["crawl_partition"])["crawl_partition"])
    chosen = sorted(choose_row_groups(scan_row_groups(), crawls, args.min_groups,
                                      args.min_docs, args.cap_per_group))
    print(f"{len(chosen)} row groups for {len(crawls)} crawls")
    rng = np.random.default_rng(args.seed)
    tables = []
    with HfFileSystem().open(DATASET, "rb", block_size=2 ** 22) as handle:
        parquet = pq.ParquetFile(handle)
        for group in chosen:
            table = parquet.read_row_group(group, columns=COLUMNS + ["source"])
            table = table.filter(pc.and_(pc.equal(table["source"], "common_crawl"),
                                         pc.is_in(table["crawl_partition"],
                                                  value_set=pa.array(sorted(crawls)))))
            frame = table.select(COLUMNS).to_pandas()
            frame = (frame.sample(frac=1, random_state=int(rng.integers(2 ** 31)))
                     .groupby("crawl_partition").head(args.cap_per_group))
            tables.append(frame.assign(row_group=group))
            print(f"  row group {group}: {len(frame):,} rows")
    out = pd.concat(tables, ignore_index=True)
    out.to_parquet(SPREAD_SAMPLE)
    print(f"wrote {SPREAD_SAMPLE}: {len(out):,} rows")



def generation_jobs(frame, per_model_reference, per_model_test, seed):
    """One seed page per domain, so no site dominates the AI reference."""
    rng = random.Random(seed)
    jobs = []
    plan = [("reference", REFERENCE_MODELS, per_model_reference),
            ("test", REFERENCE_MODELS + UNSEEN_MODELS, per_model_test)]
    for split, models, per_model in plan:
        pool = frame[frame["split"].eq("reference" if split == "reference" else "placebo")]
        pool = pool[pool["text"].map(lambda t: document_window(t) is not None)]
        pool = pool.sample(frac=1, random_state=seed).drop_duplicates("domain")
        seeds = pool.head(per_model * len(models)).to_dict("records")
        rng.shuffle(seeds)
        for i, row in enumerate(seeds):
            jobs.append({"split": split, "model": models[i % len(models)],
                         "seed_doc_id": row["doc_id"], "seed_text": main_text(row["text"])})
    return jobs


def call_openrouter(session, key, model, prompt, attempts=4):
    for attempt in range(attempts):
        response = session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            # Reasoning models otherwise spend the token budget before writing any text.
            json={"model": model, "temperature": 0.8, "max_tokens": 1200,
                  "reasoning": {"enabled": False},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120)
        if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
            time.sleep(2 ** attempt * 3)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""
    return ""


def cmd_generate(args):
    import os
    import requests
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    output = DATA / "ai_generated.jsonl"
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            done.add((row["seed_doc_id"], row["model"]))
    jobs = [job for job in generation_jobs(load_sample(), args.per_model_reference,
                                           args.per_model_test, args.seed)
            if (job["seed_doc_id"], job["model"]) not in done]
    print(f"{len(jobs)} generations to run ({len(done)} already on disk)")
    lock = threading.Lock()
    session = requests.Session()
    failures = []

    def run(job):
        seed_words = job["seed_text"].split()
        words = min(max(len(seed_words), 250), 600)
        prompt = PROMPT.format(words=words, seed=" ".join(seed_words[:350]))
        try:
            text = call_openrouter(session, key, job["model"], prompt)
        except Exception as exc:
            failures.append(f"{job['model']}: {exc}")
            return
        if not text.strip():
            failures.append(f"{job['model']}: empty response")
            return
        row = {k: job[k] for k in ("split", "model", "seed_doc_id")}
        row["text"] = text.strip()
        with lock, output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(run, jobs), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"done; {len(failures)} failures")
    for failure in failures[:10]:
        print("  ", failure)


def era_of(partition, reference_years):
    key = crawl_key(partition)
    start, end = reference_years
    if key >= CHATGPT_RELEASE:
        return "target" if key[0] >= 2023 else "transition"
    if start <= key[0] <= end:
        return "reference"
    return "placebo_forward" if key[0] > end else "placebo_backward"


def style_vocabulary(documents, allowed):
    """Words tagged mostly as adjectives or adverbs across the reference windows.
    These carry an author's habits with little topic, which is the vocabulary Liang
    et al. found most robust."""
    import nltk
    nltk.data.path.insert(0, str(DATA / "nltk_data"))
    counts = {}
    for document in documents:
        for word, tag in nltk.pos_tag(document):
            if word in allowed:
                tally = counts.setdefault(word, [0, 0])
                tally[0] += tag.startswith(("JJ", "RB"))
                tally[1] += 1
    return {word for word, (style, total) in counts.items() if style * 2 > total}


def split_ai_test(frame, seed):
    """Alternate each model's test documents between calibration and verification so
    the correction is never scored on the documents that built it."""
    halves = []
    for _, group in frame.groupby("model"):
        group = group.sample(frac=1, random_state=seed)
        halves.append(group.assign(half=np.arange(len(group)) % 2))
    frame = pd.concat(halves)
    return frame[frame["half"].eq(0)], frame[frame["half"].eq(1)]


def per_crawl(rows, args, belt):
    out = []
    for crawl, group in sorted(rows.groupby("crawl_partition"), key=lambda kv: crawl_key(kv[0])):
        if len(group) < args.min_crawl_docs:
            continue
        estimate = dict(estimate_alpha(group["ratio"].to_numpy(), resamples=args.resamples,
                                       seed=args.seed), period=crawl)
        out.append(belt.correct(estimate))
    return out


def evaluate_vocabulary(name, allowed, frame, fit_human, calibration_human, ai_reference,
                        ai_calibration, ai_verification, args):
    model = OccurrenceModel.fit(list(fit_human["window"]), list(ai_reference["window"]),
                                min_doc_frequency=args.min_doc_frequency, allowed=allowed)

    def ratios(rows):
        lh, la = model.log_likelihoods(list(rows["window"]))
        return la - lh

    frame = frame.assign(ratio=ratios(frame))
    calibration = ratios(calibration_human)
    ai_cal = ratios(ai_calibration)
    ai_ver = ratios(ai_verification)
    seen = ai_verification["model"].isin(REFERENCE_MODELS).to_numpy()
    target = frame[frame["era"].eq("target")]
    sample_size = int(target.groupby("crawl_partition").size().median())
    belt = ConfidenceBelt.build(calibration, ai_cal, sample_size,
                                repeats=args.belt_repeats, seed=args.seed)

    forward = frame[frame["era"].eq("placebo_forward")]
    backward = frame[frame["era"].eq("placebo_backward")]
    forward_rows = per_crawl(forward, args, belt)
    backward_rows = per_crawl(backward, args, belt)
    injection = {}
    for label, pool in (("seen_models", ai_ver[seen]), ("unseen_models", ai_ver[~seen])):
        for corrected in (False, True):
            injection[f"{label}{'_corrected' if corrected else ''}"] = injection_test(
                forward["ratio"].to_numpy(), pool, sample_size=sample_size, seed=args.seed,
                correct=belt.point if corrected else None)

    def verdicts(rows):
        raw = [dict(row, alpha=row["raw_alpha"], ci_low=row["raw_ci_low"]) for row in rows]
        return {"raw": placebo_verdict(raw, args.tolerance),
                "corrected": placebo_verdict(rows, args.tolerance)}

    return {
        "vocabulary": name, "vocabulary_size": len(model.vocabulary),
        "belt_sample_size": sample_size,
        "placebo_forward": {"verdict": verdicts(forward_rows), "crawls": forward_rows},
        "placebo_backward": {"verdict": verdicts(backward_rows), "crawls": backward_rows},
        "injection": injection,
        "target": per_crawl(target, args, belt),
        "top_words": model.top_words(),
    }


def cmd_evaluate(args):
    frame = load_sample()
    generated = pd.read_json(DATA / "ai_generated.jsonl", lines=True)
    rng = np.random.default_rng(args.seed)

    frame["window"] = frame["text"].map(document_window)
    frame = frame[frame["window"].notna() & ~frame["doc_id"].isin(set(generated["seed_doc_id"]))]
    frame["era"] = frame["crawl_partition"].map(lambda p: era_of(p, args.reference_years))
    generated["window"] = generated["text"].map(document_window)
    short = int(generated["window"].isna().sum())
    generated = generated[generated["window"].notna()]

    reference = frame[frame["era"].eq("reference")]
    order = rng.permutation(len(reference))
    fit_human = reference.iloc[order[:args.fit_docs]]
    calibration_human = reference.iloc[order[args.fit_docs:]]
    ai_reference = generated[generated["split"].eq("reference")]
    ai_calibration, ai_verification = split_ai_test(generated[generated["split"].eq("test")], args.seed)

    clean = dictionary_vocabulary(args.dictionary)
    vocabularies = {"full": None, "clean": clean}
    if "style" in args.vocabularies:
        tag_docs = list(fit_human["window"].head(4000)) + list(ai_reference["window"])
        vocabularies["style"] = style_vocabulary(tag_docs, clean)

    timeline = pd.read_csv(ROOT / "data" / "timeline" / "cc_crawls.csv", parse_dates=["median_date"])
    pre = timeline[timeline["median_date"] < "2022-11-30"]
    report = {
        "method": "word-occurrence mixture MLE (Liang et al. 2024), fixed 150-token prose "
                  "window, Neyman-belt bias correction",
        "eras": {"reference_years": list(args.reference_years),
                 "docs": frame["era"].value_counts().to_dict()},
        "corpus": {"human_fit_docs": len(fit_human), "human_calibration_docs": len(calibration_human),
                   "ai_reference_docs": len(ai_reference), "ai_calibration_docs": len(ai_calibration),
                   "ai_verification_docs": len(ai_verification), "ai_docs_too_short": short,
                   "reference_models": list(REFERENCE_MODELS), "unseen_models": list(UNSEEN_MODELS)},
        "iai_trend_placebo": trend_placebo(pre["median_date"], pre["median"]),
        "variants": [evaluate_vocabulary(name, vocabularies[name], frame, fit_human,
                                         calibration_human, ai_reference, ai_calibration,
                                         ai_verification, args)
                     for name in args.vocabularies],
    }
    write_json(args.output, report)
    print_report(report)


def cmd_intervals(args):
    """Replicate-bootstrap intervals and their coverage, on the spread sample."""
    frame = pd.read_parquet(SPREAD_SAMPLE).drop_duplicates("doc_id")
    generated = pd.read_json(DATA / "ai_generated.jsonl", lines=True)
    rng = np.random.default_rng(args.seed)
    frame["window"] = frame["text"].map(document_window)
    frame = frame[frame["window"].notna() & ~frame["doc_id"].isin(set(generated["seed_doc_id"]))]
    frame["era"] = frame["crawl_partition"].map(lambda p: era_of(p, args.reference_years))
    frame = frame.reset_index(drop=True)
    generated["window"] = generated["text"].map(document_window)
    generated = generated[generated["window"].notna()]

    reference = frame[frame["era"].eq("reference")]
    order = rng.permutation(len(reference))
    fit_human = reference.iloc[order[:args.fit_docs]]
    calibration_human = reference.iloc[order[args.fit_docs:]]
    ai_reference = generated[generated["split"].eq("reference")]
    ai_calibration, ai_verification = split_ai_test(generated[generated["split"].eq("test")], args.seed)

    model = OccurrenceModel.fit(list(fit_human["window"]), list(ai_reference["window"]),
                                min_doc_frequency=args.min_doc_frequency,
                                allowed=dictionary_vocabulary(args.dictionary))
    matrix = model.matrix(list(frame["window"]))
    pools = {name: model.matrix(list(rows["window"])) for name, rows in (
        ("fit_human", fit_human), ("fit_ai", ai_reference),
        ("calibration_human", calibration_human), ("calibration_ai", ai_calibration),
        ("verification_ai", ai_verification))}

    measured = frame[frame["era"].isin(["placebo_backward", "placebo_forward", "target"])]
    periods = {}
    for crawl, group in measured.groupby("crawl_partition"):
        if len(group) >= args.min_crawl_docs:
            periods[crawl] = (matrix[group.index.to_numpy()], group["row_group"].to_numpy())
    sample_size = int(np.median([m.shape[0] for m, _ in periods.values()]))

    # Coverage check: real pre-ChatGPT crawls with a known share of their documents
    # swapped for held-out AI text. The crawl keeps its own composition and clusters.
    injected = {}
    placebo_crawls = sorted((c for c in periods if era_of(c, args.reference_years) != "target"),
                            key=crawl_key)
    ai_rows = pools["verification_ai"]
    for crawl in placebo_crawls[::max(1, len(placebo_crawls) // args.coverage_crawls)]:
        crawl_matrix, clusters = periods[crawl]
        for share in args.injection_shares:
            k = int(round(share * crawl_matrix.shape[0]))
            swap = rng.choice(crawl_matrix.shape[0], k, replace=False)
            keep = np.setdiff1d(np.arange(crawl_matrix.shape[0]), swap)
            ai_pick = rng.choice(ai_rows.shape[0], k, replace=k > ai_rows.shape[0])
            injected[f"{crawl}+{share:.2f}"] = (
                sparse.vstack([crawl_matrix[keep], ai_rows[ai_pick]]).tocsr(),
                np.r_[clusters[keep], clusters[swap]], share)

    belt = ConfidenceBelt.build(model.log_ratio_matrix(pools["calibration_human"]),
                                model.log_ratio_matrix(pools["calibration_ai"]),
                                sample_size, repeats=args.belt_repeats, seed=args.seed)
    intervals = replicate_intervals(
        model, pools["fit_human"], pools["fit_ai"], pools["calibration_human"],
        pools["calibration_ai"],
        {**periods, **{name: (m, c) for name, (m, c, _) in injected.items()}},
        sample_size, replicates=args.replicates, belt_repeats=args.replicate_belt_repeats,
        seed=args.seed)

    def row(name, crawl_matrix, clusters):
        raw = estimate_alpha(model.log_ratio_matrix(crawl_matrix), resamples=args.resamples,
                             seed=args.seed)
        doc_only = belt.correct(raw)
        return {"period": name, "n": int(crawl_matrix.shape[0]),
                "row_groups": int(np.unique(clusters).size),
                "raw_alpha": raw["alpha"], "alpha": doc_only["alpha"],
                "doc_bootstrap_ci": [doc_only["ci_low"], doc_only["ci_high"]],
                "ci": intervals[name]["ci"], "raw_ci": intervals[name]["raw_ci"]}

    rows = [dict(row(c, *periods[c]), era=era_of(c, args.reference_years))
            for c in sorted(periods, key=crawl_key)]
    injection_rows = [dict(row(name, m, c), true_alpha=share)
                      for name, (m, c, share) in injected.items()]

    def covered(rows, truth):
        hits = [r["ci"][0] - 1e-9 <= truth(r) <= r["ci"][1] + 1e-9 for r in rows]
        return {"covered": int(sum(hits)), "total": len(hits),
                "rate": float(np.mean(hits)) if hits else None}

    placebo_rows = [r for r in rows if r["era"] != "target"]
    report = {
        "method": "clean-vocabulary mixture MLE; intervals from replicate bootstrap over "
                  "reference corpora, calibration pools and (row group, document) crawl resamples",
        "replicates": args.replicates, "belt_sample_size": sample_size,
        "vocabulary": len(model.vocabulary),
        "coverage": {"placebo_contains_zero": covered(placebo_rows, lambda r: 0.0),
                     "injected_contains_truth": covered(injection_rows, lambda r: r["true_alpha"])},
        "crawls": rows, "injected": injection_rows,
    }
    write_json(args.output, report)

    def fmt(r):
        return (f"  {r['period']:16} n={r['n']:>5,} rg={r['row_groups']}  raw {r['raw_alpha']:.3f}  "
                f"corrected {r['alpha']:.3f}  doc-only [{r['doc_bootstrap_ci'][0]:.3f}, "
                f"{r['doc_bootstrap_ci'][1]:.3f}]  replicate [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
    print(f"vocabulary {len(model.vocabulary):,}  replicates {args.replicates}  belt n {sample_size:,}")
    for era in ("placebo_backward", "placebo_forward", "target"):
        print(f"\n{era}")
        for r in rows:
            if r["era"] == era:
                print(fmt(r))
    print("\ninjected (true share after +)")
    for r in injection_rows:
        print(fmt(r))
    print("\ncoverage:", json.dumps(report["coverage"]))


def _fmt(row, key="alpha"):
    return f"{row[key]:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def print_report(report):
    trend = report["iai_trend_placebo"]
    print(f"IAI trend placebo: slope {trend['slope_per_year']:+.2f}/yr "
          f"[{trend['ci_low']:+.2f}, {trend['ci_high']:+.2f}] -> {'PASS' if trend['passed'] else 'FAIL'}")
    print("Eras:", report["eras"])
    print("Corpus:", {k: v for k, v in report["corpus"].items() if "models" not in k})
    for variant in report["variants"]:
        print(f"\n===== vocabulary={variant['vocabulary']} ({variant['vocabulary_size']:,} words, "
              f"belt n={variant['belt_sample_size']:,})")
        for side in ("placebo_forward", "placebo_backward"):
            v = variant[side]["verdict"]
            print(f"{side}: raw {'PASS' if v['raw']['passed'] else 'FAIL'} mean {v['raw']['mean_alpha']:.3f}"
                  f" | corrected {'PASS' if v['corrected']['passed'] else 'FAIL'} mean {v['corrected']['mean_alpha']:.3f}")
            for row in variant[side]["crawls"]:
                print(f"  {row['period']}  raw {row['raw_alpha']:.3f}  corrected {_fmt(row)}  n={row['n']:,}")
        for name, rows in variant["injection"].items():
            print(f"injection {name:24}" + "  ".join(
                f"{r['true_alpha']:.2f}->{r['mean_estimate']:.3f}" for r in rows)
                + f"   rmse {np.mean([r['rmse'] for r in rows]):.3f}")
        print("target:")
        for row in variant["target"]:
            print(f"  {row['period']}  raw {row['raw_alpha']:.3f}  corrected {_fmt(row)}  n={row['n']:,}")
        print("AI-leaning:", ", ".join(variant["top_words"]["ai"]))
        print("human-leaning:", ", ".join(variant["top_words"]["human"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sample")
    spread = commands.add_parser("sample-spread")
    spread.add_argument("--min-groups", type=int, default=3)
    spread.add_argument("--min-docs", type=int, default=2000)
    spread.add_argument("--cap-per-group", type=int, default=700)
    spread.add_argument("--seed", type=int, default=0)
    generate = commands.add_parser("generate")
    generate.add_argument("--per-model-reference", type=int, default=300)
    generate.add_argument("--per-model-test", type=int, default=90)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--seed", type=int, default=0)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--output", default=str(DATA / "report.json"))
    evaluate.add_argument("--fit-docs", type=int, default=15000)
    evaluate.add_argument("--reference-years", type=int, nargs=2, default=(2019, 2021))
    evaluate.add_argument("--vocabularies", nargs="+", default=["full", "clean", "style"],
                          choices=["full", "clean", "style"])
    evaluate.add_argument("--dictionary", default="/usr/share/dict/american-english")
    evaluate.add_argument("--belt-repeats", type=int, default=200)
    evaluate.add_argument("--min-doc-frequency", type=float, default=0.01)
    evaluate.add_argument("--min-crawl-docs", type=int, default=300)
    evaluate.add_argument("--resamples", type=int, default=200)
    evaluate.add_argument("--tolerance", type=float, default=0.02)
    evaluate.add_argument("--seed", type=int, default=0)
    intervals = commands.add_parser("intervals")
    intervals.add_argument("--output", default=str(DATA / "intervals.json"))
    intervals.add_argument("--fit-docs", type=int, default=15000)
    intervals.add_argument("--reference-years", type=int, nargs=2, default=(2019, 2021))
    intervals.add_argument("--dictionary", default="/usr/share/dict/american-english")
    intervals.add_argument("--min-doc-frequency", type=float, default=0.01)
    intervals.add_argument("--min-crawl-docs", type=int, default=300)
    intervals.add_argument("--replicates", type=int, default=40)
    intervals.add_argument("--belt-repeats", type=int, default=200)
    intervals.add_argument("--replicate-belt-repeats", type=int, default=60)
    intervals.add_argument("--resamples", type=int, default=200)
    intervals.add_argument("--coverage-crawls", type=int, default=8)
    intervals.add_argument("--injection-shares", type=float, nargs="+", default=[0.03, 0.08])
    intervals.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"sample": cmd_sample, "sample-spread": cmd_sample_spread, "generate": cmd_generate,
     "evaluate": cmd_evaluate, "intervals": cmd_intervals}[args.command](args)


if __name__ == "__main__":
    main()

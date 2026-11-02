"""
Push Silver and Gold Parquet datasets to Hugging Face.

Called by the pipeline GitHub Actions workflow after each index update.
Requires HF_TOKEN env var (write-access token from huggingface.co/settings/tokens).

Usage:
    python scripts/push_to_hf.py --repo jupiternull/dead-internet-observatory
"""

import argparse
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError


STATE_FILES = (
    "gold/doc_registry.parquet",
    "gold/scored.parquet",
    "observatory.db",
)
DASHBOARD_DB = "dashboard.db"

# observatory.db carries the document ledger and per-domain scores, which makes it
# grow into the hundreds of megabytes. The dashboard only reads the aggregate
# tables below, so it gets a compact copy and never downloads the full snapshot.
DASHBOARD_TABLES = (
    "composite_index",
    "daily_index",
    "index_quality",
    "source_interval",
    "meta",
)
RESTORE_RETRY_STATUSES = {429, 500, 502, 503, 504}


def export_dashboard_db(source: Path, destination: Path) -> Path:
    """Copy the dashboard's aggregate tables into a compact SQLite file."""
    if destination.exists():
        destination.unlink()
    with sqlite3.connect(str(destination)) as out:
        out.execute("ATTACH DATABASE ? AS src", (str(source),))
        for table in DASHBOARD_TABLES:
            row = out.execute(
                "SELECT sql FROM src.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if row is None:
                continue
            out.execute(row[0])
            out.execute(f'INSERT INTO "{table}" SELECT * FROM src."{table}"')
        out.commit()
    return destination


def is_rate_limit(exc):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if response is not None and response.status_code == 429:
            return True
        text = str(current).lower()
        if "429" in text or "too many requests" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def is_transient_hf_error(exc):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if response is not None and response.status_code in RESTORE_RETRY_STATUSES:
            return True
        current = current.__cause__ or current.__context__
    return is_rate_limit(exc)


def retry_hf(label, func, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if is_rate_limit(exc):
                if attempt == max_attempts:
                    raise
                delay = 60 * attempt
                print(f"[HF] Rate limited while {label}; retrying in {delay}s")
                time.sleep(delay)
                continue
            raise


def restore(repo_id: str, data_root: Path):
    token = os.environ.get("HF_TOKEN", "")
    restored = 0

    for relative_path in STATE_FILES:
        destination = data_root / relative_path

        for attempt in range(1, 4):
            try:
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=relative_path,
                    repo_type="dataset",
                    token=token or None,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached_path, destination)
                restored += 1
                print(f"[HF] Restored {relative_path}")
                break
            except RemoteEntryNotFoundError:
                print(f"[HF] No prior {relative_path}")
                break
            except Exception as exc:
                if attempt == 3:
                    if is_transient_hf_error(exc):
                        if destination.exists():
                            print(
                                f"[HF] Restore deferred for {relative_path}: {exc}; "
                                "using cached local state"
                            )
                        else:
                            print(
                                f"[HF] Restore deferred for {relative_path}: {exc}; "
                                "state file unavailable"
                            )
                        break
                    print(f"[HF] Restore failed for {relative_path}: {exc}")
                    raise
                delay = 30 * attempt
                print(f"[HF] Restore {relative_path} failed: {exc}; retrying in {delay}s")
                time.sleep(delay)

    print(f"[HF] Restored {restored}/{len(STATE_FILES)} state files")


def push(repo_id: str, data_root: Path, token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed — run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    parquet_files = sorted(data_root.glob("silver/**/*.parquet"))
    parquet_files.extend(sorted(data_root.glob("gold/**/*.parquet")))
    db_path = data_root / "observatory.db"
    dashboard_path = data_root / DASHBOARD_DB
    if db_path.exists():
        export_dashboard_db(db_path, dashboard_path)
        print(
            f"[HF] Dashboard snapshot: {dashboard_path.stat().st_size / 1e6:.1f} MB "
            f"(from {db_path.stat().st_size / 1e6:.0f} MB)"
        )
    uploaded = len(parquet_files) + int(db_path.exists()) + int(dashboard_path.exists())
    if not uploaded:
        print("[HF] Error: no Parquet datasets or observatory.db found")
        sys.exit(1)

    print(f"[HF] Uploading {uploaded} dataset files in one commit …")
    try:
        retry_hf(
            "uploading dataset snapshot",
            lambda: api.upload_folder(
                folder_path=str(data_root),
                path_in_repo="",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                allow_patterns=["*.parquet", "observatory.db", DASHBOARD_DB],
                commit_message="update observatory dataset snapshot",
            ),
        )
    except Exception as exc:
        if not is_rate_limit(exc):
            print(f"[HF] Dataset upload failed: {exc}")
            sys.exit(1)
        print("[HF] Dataset sync deferred due to rate limiting.")
        print("[HF] Pipeline output is complete; Hugging Face sync can catch up on the next run.")
        return

    print(f"[HF] ✓ Done — {uploaded} files pushed to {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true",
                        help="Restore persistent pipeline state from Hugging Face")
    parser.add_argument("--repo", default="jupiternull/dead-internet-observatory",
                        help="HuggingFace dataset repo ID (owner/name)")
    parser.add_argument("--data-root", default="./data",
                        help="Path to the local data/ directory")
    args = parser.parse_args()

    if args.restore:
        restore(args.repo, Path(args.data_root))
        sys.exit(0)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    push(args.repo, Path(args.data_root), token)

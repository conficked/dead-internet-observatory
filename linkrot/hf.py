"""Small Hugging Face helpers for the census workflow.

    python -m linkrot.hf get linkrot/sample.jsonl.gz sample.jsonl.gz   # exit 3 if missing
    python -m linkrot.hf put sample.jsonl.gz linkrot/sample.jsonl.gz "message"
"""

import os
import shutil
import sys
import time

REPO = os.environ.get("HF_REPO", "conficked/dead-internet-observatory")


def get(remote: str, local: str) -> int:
    from huggingface_hub import errors, hf_hub_download

    missing = tuple(getattr(errors, n) for n in ("EntryNotFoundError", "RemoteEntryNotFoundError")
                    if hasattr(errors, n))

    local_dir = os.path.dirname(os.path.abspath(local))
    os.makedirs(local_dir, exist_ok=True)
    for attempt in range(3):
        try:
            path = hf_hub_download(REPO, remote, repo_type="dataset", local_dir=local_dir,
                                   token=os.environ.get("HF_TOKEN") or None)
        except missing:
            print(f"[hf] {remote} not found in {REPO}")
            return 3
        except Exception as exc:  # transient network or rate limit
            if attempt == 2:
                raise
            print(f"[hf] download failed ({exc}); retrying")
            time.sleep(30 * (attempt + 1))
            continue
        if os.path.abspath(path) != os.path.abspath(local):
            shutil.move(path, local)
        print(f"[hf] {remote} -> {local} ({os.path.getsize(local) / 1e6:.1f} MB)")
        return 0
    return 1


def put(local: str, remote: str, message: str) -> int:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[hf] HF_TOKEN not set; skipping upload")
        return 0
    for attempt in range(3):
        try:
            HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO,
                                repo_type="dataset", token=token, commit_message=message)
            print(f"[hf] {local} -> {REPO}/{remote}")
            return 0
        except Exception as exc:
            if attempt == 2:
                print(f"[hf] upload failed: {exc}")
                return 1
            time.sleep(30 * (attempt + 1))
    return 1


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "get":
        sys.exit(get(*args))
    if cmd == "put":
        sys.exit(put(args[0], args[1], args[2] if len(args) > 2 else f"linkrot: update {args[1]}"))
    sys.exit(f"unknown command {cmd}")

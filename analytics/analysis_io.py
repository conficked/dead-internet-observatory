"""Shared JSON writing for the IAI v2 analysis scripts."""

import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "iai-v2-analysis-1"


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "item") and callable(value.item):
        return _clean(value.item())
    return value


def write_json(path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(payload), indent=2, sort_keys=True, allow_nan=False)
    destination.write_text(text + "\n")

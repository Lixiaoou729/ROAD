"""Content identities and atomic metadata writes; no training/model imports."""

import hashlib
import json
import os
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def object_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def asset_fingerprint(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Model asset directory missing: {root}")
    # Include tokenizer merges/vocab and converted metric checkpoints as well as
    # the normal HF files. Ignore only non-runtime download bookkeeping.
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".pth", ".pt", ".tiktoken"}
    entries = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes and ".cache" not in path.relative_to(root).parts
    }
    if not entries:
        raise ValueError("No model asset files found")
    # Keep the v1 serialization so unmodified native snapshots retain identity.
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        # Hard-link publication is atomic and cannot clobber another writer.
        # Both names are on the same filesystem; retain partial files on error.
        os.link(temporary, path)
        temporary.unlink()


def checked_files(root, files):
    """Validate an explicit file inventory, rejecting traversal and symlinks out."""
    root = Path(root).resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing artifact file inventory")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Artifact content mismatch: {name}")

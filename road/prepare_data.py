"""Index prepared RGB/caption tar shards without extracting or changing data."""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

from .artifacts import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    sidecar = out.with_suffix(".assets.json")
    if out.exists() or sidecar.exists():
        raise FileExistsError("Use a new manifest and sidecar destination")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, shards = [], {}
    for shard in sorted(Path(name).resolve() for name in args.shards):
        relative = os.path.relpath(shard, out.parent)
        if relative in shards:
            raise ValueError(f"Repeated shard: {shard.name}")
        sha = digest(shard)
        shards[relative] = sha
        before = len(rows)
        with tarfile.open(shard) as tar:
            plain = isinstance(tar.fileobj, io.BufferedReader)
            samples = {}
            for member in tar:
                if not member.isfile():
                    continue
                key, extension = os.path.splitext(member.name)
                extension = extension.lower()
                if extension not in (".jpg", ".jpeg", ".png", ".webp", ".txt", ".json"):
                    continue
                sample = samples.setdefault(key, {})
                if extension in (".jpg", ".jpeg", ".png", ".webp"):
                    if "member" in sample:
                        raise ValueError(f"Duplicate image key: {key}")
                    sample["member"] = member.name
                    if plain:
                        sample.update(offset=member.offset_data, size=member.size)
                else:
                    if extension in sample:
                        raise ValueError(f"Duplicate caption record: {key}{extension}")
                    with tar.extractfile(member) as stream:
                        if extension == ".txt":
                            sample[extension] = stream.read().decode("utf-8").strip()
                        else:
                            metadata = json.load(stream)
                            sample[extension] = metadata.get("caption", metadata.get("text"))
            for key, sample in samples.items():
                if "member" not in sample:
                    continue
                # TXT takes precedence over JSON regardless of tar member order.
                caption = sample.pop(".txt", sample.get(".json"))
                sample.pop(".json", None)
                if not isinstance(caption, str):
                    raise ValueError(f"Image lacks a string caption: {key}")
                rows.append({
                    "id": sha + ":" + key, "image": relative,
                    "caption": caption, **sample,
                })
        if len(rows) == before:
            raise ValueError(f"Shard has no image-caption pairs: {shard.name}")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty dataset or repeated sample IDs/shard contents")
    temporary = out.with_name(out.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(sidecar, {
        "shards": shards, "count": len(rows), "manifest_sha256": digest(temporary),
    })
    os.link(temporary, out)
    temporary.unlink()
    print(f"Indexed {len(rows)} pairs; this does not establish paper-pool identity.")


if __name__ == "__main__":
    main()

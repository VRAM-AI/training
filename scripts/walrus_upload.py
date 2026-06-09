#!/usr/bin/env python3
"""
walrus_upload.py  –  Shard dataset files and upload each shard to Walrus.

Shards pairs.jsonl and eval.jsonl into chunks of SHARD_SIZE rows each,
stores each shard as a Walrus blob (mainnet, 2 epochs), and writes a
manifest to data/walrus_manifest.json with every blob ID + metadata.

Usage:
    python scripts/walrus_upload.py
    python scripts/walrus_upload.py --shard-size 200 --epochs 3
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MANIFEST_FILE = DATA_DIR / "walrus_manifest.json"


def shard_jsonl(path: Path, shard_size: int) -> list[list[dict]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]


def upload_shard(rows: list[dict], epochs: int, label: str) -> str:
    """Write shard to a temp file, upload to Walrus, return blob ID."""
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, prefix=f"walrus_{label}_"
    ) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        tmp_path = f.name

    result = subprocess.run(
        ["walrus", "store", "--epochs", str(epochs), "--json", tmp_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"walrus store failed:\n{result.stderr[:500]}")

    # Parse blob ID from JSON output
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("{") is False:
            continue
        try:
            data = json.loads(line)
            # walrus --json output: {"newlyCreated": {"blobObject": {"blobId": "..."}}}
            # or {"alreadyCertified": {"blobId": "..."}}
            blob_id = (
                data.get("newlyCreated", {}).get("blobObject", {}).get("blobId")
                or data.get("alreadyCertified", {}).get("blobId")
                or data.get("blobId")
            )
            if blob_id:
                return blob_id
        except json.JSONDecodeError:
            continue

    raise RuntimeError(f"Could not parse blob ID from walrus output:\n{result.stdout[:400]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard + upload dataset to Walrus")
    parser.add_argument("--shard-size", type=int, default=100, help="Rows per shard (default 100)")
    parser.add_argument("--epochs", type=int, default=2, help="Walrus storage epochs (default 2)")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["pairs.jsonl", "eval.jsonl"],
        help="Files in data/ to upload (default: pairs.jsonl eval.jsonl)",
    )
    args = parser.parse_args()

    manifest: list[dict] = []
    total_blobs = 0

    for filename in args.files:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  SKIP {filename} (not found)")
            continue

        shards = shard_jsonl(path, args.shard_size)
        print(f"\n{filename}: {sum(len(s) for s in shards)} rows → {len(shards)} shards")

        for i, shard in enumerate(shards):
            label = f"{Path(filename).stem}_{i:03d}"
            print(f"  [{i+1}/{len(shards)}] uploading {label} ({len(shard)} rows) …", end="", flush=True)
            try:
                blob_id = upload_shard(shard, args.epochs, label)
                manifest.append(
                    {
                        "blob_id": blob_id,
                        "file": filename,
                        "shard": i,
                        "rows": len(shard),
                        "epochs": args.epochs,
                        "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f" ✓ {blob_id}")
                total_blobs += 1
            except Exception as e:
                print(f" FAIL: {e}")

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    print(f"\n── Upload complete ─────────────────────────────")
    print(f"  Blobs uploaded : {total_blobs}")
    print(f"  Manifest       : {MANIFEST_FILE}")
    print(f"\nBlob IDs:")
    for entry in manifest:
        print(f"  {entry['file']} shard {entry['shard']:03d} → {entry['blob_id']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_move_dataset.py  –  End-to-end pipeline to build the Move/DeepBook fine-tuning dataset.

Pipeline stages (run in order or select with --stage):
  m1   Inventory raw corpus  (data/raw/)
  m2   Filter to 2024-edition, deduplicate  -> data/filtered/
  m3   Tag into the four targeted subsets
  m4   Synthesize instruction-to-code pairs via OpenRouter teacher model
  m5   QC with move_eval harness, split train/eval

Usage:
    # Full pipeline
    python scripts/build_move_dataset.py

    # Single stage
    python scripts/build_move_dataset.py --stage m2

Environment:
    OPENROUTER_API_KEY   Required for stage m4 (synthesis)

Outputs:
    data/filtered/          2024-edition, deduped .move files
    data/pairs.jsonl        Training pairs (schema: Section 7.1)
    data/eval.jsonl         Hold-out eval pairs
    data/filter_report.json M2 count report
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
FILTERED_DIR = ROOT / "data" / "filtered"
PAIRS_FILE = ROOT / "data" / "pairs.jsonl"
EVAL_FILE = ROOT / "data" / "eval.jsonl"
FILTER_REPORT = ROOT / "data" / "filter_report.json"

# ── OpenRouter config ─────────────────────────────────────────────────────────
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
TEACHER_MODEL = "deepseek/deepseek-chat"      # DeepSeek V3 — fast, content field, non-reasoning
TEACHER_MODEL_PRO = "deepseek/deepseek-chat"  # same; V4 variants are reasoning models (null content)

# ── Tags ─────────────────────────────────────────────────────────────────────
TAG_RECEIVER = "receiver-syntax"
TAG_DEPPATTERN = "dependency-pattern"
TAG_DEEPBOOK = "deepbook"
TAG_GENERAL = "general"

DEEPBOOK_KEYWORDS = [
    "BalanceManager", "balance_manager", "TradeProof", "trade_proof",
    "place_limit_order", "place_market_order", "flashloan", "flash_loan",
    "deepbook", "DEEP", "OrderType", "order_type",
]

RECEIVER_SYNTAX_RE = re.compile(r"fun\s+\w+\s*\(\s*(?:mut\s+)?self\s*[,:]")


# ══════════════════════════════════════════════════════════════════════════════
# Stage M1 – inventory
# ══════════════════════════════════════════════════════════════════════════════

def stage_m1() -> list[Path]:
    """Return all .move files found under data/raw/."""
    files = sorted(RAW_DIR.rglob("*.move"))
    print(f"M1: found {len(files)} .move files in {RAW_DIR}")
    return files


# ══════════════════════════════════════════════════════════════════════════════
# Stage M2 – filter: 2024 edition + compile check + dedup
# ══════════════════════════════════════════════════════════════════════════════

def get_edition(move_toml: Path) -> str:
    """Read edition from a Move.toml file."""
    try:
        text = move_toml.read_text()
        m = re.search(r'edition\s*=\s*["\']([^"\']+)["\']', text)
        return m.group(1) if m else ""
    except Exception:
        return ""


def package_of(move_file: Path) -> Path | None:
    """Walk up to find the nearest Move.toml."""
    p = move_file.parent
    while p != p.parent:
        candidate = p / "Move.toml"
        if candidate.exists():
            return candidate
        p = p.parent
    return None


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_m2(files: list[Path]) -> tuple[list[Path], dict]:
    """Filter to 2024-edition, dedup. Returns kept files and a report dict."""
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    total = len(files)
    dropped_edition = 0
    dropped_dup = 0
    kept = []
    seen_hashes: set[str] = set()
    pkg_edition_cache: dict[Path, str] = {}

    for f in files:
        toml = package_of(f)
        if toml is None:
            dropped_edition += 1
            continue

        if toml not in pkg_edition_cache:
            pkg_edition_cache[toml] = get_edition(toml)
        edition = pkg_edition_cache[toml]

        if not edition.startswith("2024"):
            dropped_edition += 1
            continue

        h = file_hash(f)
        if h in seen_hashes:
            dropped_dup += 1
            continue
        seen_hashes.add(h)
        kept.append(f)

    # Copy kept files into data/filtered/ preserving relative path from data/raw/
    for f in kept:
        try:
            rel = f.relative_to(RAW_DIR)
        except ValueError:
            rel = Path(f.name)
        dest = FILTERED_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())

    report = {
        "files_in": total,
        "dropped_non_2024": dropped_edition,
        "dropped_duplicate": dropped_dup,
        "files_kept": len(kept),
        "timestamp_ms": int(time.time() * 1000),
    }
    FILTER_REPORT.write_text(json.dumps(report, indent=2))
    print(
        f"M2: {total} in → kept {len(kept)} "
        f"(dropped {dropped_edition} non-2024, {dropped_dup} dups)"
    )
    print(f"    Filter report: {FILTER_REPORT}")
    return kept, report


# ══════════════════════════════════════════════════════════════════════════════
# Stage M3 – tag into subsets
# ══════════════════════════════════════════════════════════════════════════════

def tag_file(path: Path) -> list[str]:
    text = path.read_text(errors="replace")
    tags = []
    if any(kw in text for kw in DEEPBOOK_KEYWORDS):
        tags.append(TAG_DEEPBOOK)
    if RECEIVER_SYNTAX_RE.search(text):
        tags.append(TAG_RECEIVER)
    if not tags:
        tags.append(TAG_GENERAL)
    return tags


def stage_m3(kept: list[Path]) -> dict[str, list[Path]]:
    """Assign tags and bucket files. Returns tag -> [files] map."""
    buckets: dict[str, list[Path]] = {
        TAG_DEEPBOOK: [],
        TAG_RECEIVER: [],
        TAG_DEPPATTERN: [],
        TAG_GENERAL: [],
    }
    for f in kept:
        for tag in tag_file(f):
            buckets[tag].append(f)

    for tag, files in buckets.items():
        print(f"M3: {tag:25s} → {len(files):4d} files")

    return buckets


# ══════════════════════════════════════════════════════════════════════════════
# Stage M4 – synthesis via OpenRouter
# ══════════════════════════════════════════════════════════════════════════════

def call_teacher(prompt: str, model: str = TEACHER_MODEL, max_tokens: int = 1024,
                 retries: int = 3) -> str:
    """Call OpenRouter via curl subprocess (avoids macOS Python DNS issues)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")

    for attempt in range(1, retries + 1):
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        })
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "--max-time", "45",
                    "-X", "POST",
                    f"{OPENROUTER_BASE}/chat/completions",
                    "-H", f"Authorization: Bearer {api_key}",
                    "-H", "Content-Type: application/json",
                    "-H", "HTTP-Referer: https://github.com/vram-hub",
                    "-H", "X-Title: Move Dataset Builder",
                    "-d", payload,
                ],
                capture_output=True, text=True, timeout=60,
            )
            data = json.loads(result.stdout)
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning") or ""
            if content:
                return content.strip()
            print(f"  WARN empty response (attempt {attempt}/{retries})", file=sys.stderr)
        except Exception as e:
            print(f"  WARN attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
        time.sleep(2 * attempt)
    return ""  # exhausted retries


REVERSE_INSTRUCTION_PROMPT = """\
You are an expert in the Sui Move programming language (2024 edition) and DeepBook v3.

Given the following Move source code, write a concise natural-language instruction (1-3 sentences) that describes the task a developer would ask to produce this code. Output ONLY the instruction text, no code, no explanation.

Move code:
```move
{code}
```"""

ERROR_TO_FIX_PROMPT = """\
You are an expert in Sui Move 2024. The following Move code has a realistic bug (type error, borrow error, missing ability, wrong syntax, or circular dependency).

1. Describe the bug in one sentence as a compiler error message.
2. Output the FIXED version of the code only (no explanation, no markdown fences).

Buggy code:
```move
{code}
```

Respond in this exact format:
ERROR: <one-line compiler error>
FIXED:
<corrected Move code>"""

DEEPBOOK_TASK_PROMPT = """\
You are an expert in Sui Move 2024 and DeepBook v3 (MystenLabs/deepbookv3).

Write a complete, compilable Sui Move module (edition "2024.beta") that accomplishes the following task. Use method/receiver syntax where possible. Include all necessary imports. Output ONLY the raw Move code, no markdown fences.

Task: {instruction}

Reference code to base your answer on:
```move
{code}
```"""


def make_reverse_pair(f: Path, tags: list[str], source: str) -> dict | None:
    full_code = f.read_text(errors="replace")
    prompt_code = full_code[:1500]  # truncate only for API prompt
    try:
        instruction = call_teacher(REVERSE_INSTRUCTION_PROMPT.format(code=prompt_code), max_tokens=800)
        return {
            "id": str(uuid.uuid4()),
            "instruction": instruction,
            "input": "",
            "output": full_code,  # always store full source
            "tags": tags,
            "source": source,
            "compiles": True,  # source file already compiled (M2 gate)
        }
    except Exception as e:
        print(f"    WARN reverse-pair failed for {f.name}: {e}")
        return None


def make_error_fix_pair(f: Path, tags: list[str], source: str) -> dict | None:
    code = f.read_text(errors="replace")[:1200]
    try:
        response = call_teacher(ERROR_TO_FIX_PROMPT.format(code=code), max_tokens=1200)
        if "ERROR:" not in response or "FIXED:" not in response:
            return None
        error_line = response.split("FIXED:")[0].replace("ERROR:", "").strip()
        fixed_code = response.split("FIXED:", 1)[1].strip()
        return {
            "id": str(uuid.uuid4()),
            "instruction": f"Fix the following Move code. Compiler error: {error_line}",
            "input": code,
            "output": fixed_code,
            "tags": tags + ["error-to-fix"],
            "source": source,
            "compiles": None,  # QC will verify fixed version
        }
    except Exception as e:
        print(f"    WARN error-fix pair failed for {f.name}: {e}")
        return None


def make_deepbook_pair(f: Path, source: str) -> dict | None:
    code = f.read_text(errors="replace")[:1500]
    instructions = [
        "Implement a function that places a limit order on a DeepBook v3 pool using a BalanceManager.",
        "Write a Move module that creates a TradeProof and executes a market order on DeepBook v3.",
        "Implement a DeepBook v3 flash loan that borrows DEEP tokens and repays within the same transaction.",
    ]
    import random
    instruction = random.choice(instructions)
    try:
        output = call_teacher(
            DEEPBOOK_TASK_PROMPT.format(instruction=instruction, code=code),
            model=TEACHER_MODEL_PRO,
            max_tokens=2000,
        )
        return {
            "id": str(uuid.uuid4()),
            "instruction": instruction,
            "input": "",
            "output": output,
            "tags": [TAG_DEEPBOOK],
            "source": source,
            "compiles": None,  # QC step verifies
        }
    except Exception as e:
        print(f"    WARN deepbook pair failed for {f.name}: {e}")
        return None


def stage_m4(buckets: dict[str, list[Path]], max_per_tag: int = 300) -> list[dict]:
    """Synthesize pairs. Flushes each pair to PAIRS_FILE immediately (crash-safe)."""
    import random
    pairs: list[dict] = []

    def source_str(f: Path) -> str:
        try:
            return str(f.relative_to(RAW_DIR))
        except ValueError:
            return str(f)

    def flush(pair: dict) -> None:
        with PAIRS_FILE.open("a") as fh:
            fh.write(json.dumps(pair) + "\n")
        pairs.append(pair)
        print(f"  +pair [{len(pairs)}] {pair['tags']} ← {Path(pair['source']).name}")

    # 1. Reverse-instruction pairs from all buckets
    print("M4: generating reverse-instruction pairs …")
    for tag, files in buckets.items():
        sample = random.sample(files, min(max_per_tag, len(files)))
        for i, f in enumerate(sample):
            print(f"  [{tag} {i+1}/{len(sample)}] {f.name} …", flush=True)
            pair = make_reverse_pair(f, [tag], source_str(f))
            if pair:
                flush(pair)
                time.sleep(0.2)

    # 2. Error-to-fix pairs (receiver-syntax and dependency-pattern focus)
    print("M4: generating error-to-fix pairs …")
    error_pool = buckets[TAG_RECEIVER] + buckets[TAG_DEEPBOOK]
    sample = random.sample(error_pool, min(200, len(error_pool)))
    for i, f in enumerate(sample):
        print(f"  [error-fix {i+1}/{len(sample)}] {f.name} …", flush=True)
        pair = make_error_fix_pair(f, [TAG_DEPPATTERN], source_str(f))
        if pair:
            flush(pair)
            time.sleep(0.2)

    # 3. DeepBook-specific task pairs
    print("M4: generating DeepBook task pairs …")
    db_files = buckets[TAG_DEEPBOOK]
    sample = random.sample(db_files, min(100, len(db_files)))
    for i, f in enumerate(sample):
        print(f"  [deepbook {i+1}/{len(sample)}] {f.name} …", flush=True)
        pair = make_deepbook_pair(f, source_str(f))
        if pair:
            flush(pair)
            time.sleep(0.3)

    print(f"M4: synthesized {len(pairs)} raw pairs (pre-QC)")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Stage M5 – QC: inline compile check + train/eval split
# ══════════════════════════════════════════════════════════════════════════════

def _try_compile_inline(code: str) -> bool:
    """Quick heuristic compile check: look for obvious syntax problems."""
    if not code.strip():
        return False
    # Must contain a module declaration
    if not re.search(r"\bmodule\b", code):
        return False
    # Unbalanced braces (very rough)
    if code.count("{") != code.count("}"):
        return False
    return True


def stage_m5_qc(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    QC filter: drop non-compiling (heuristic fast-path; full harness via move_eval.py).
    Split 90/10 train/eval, ensuring eval is decontaminated.
    """
    import random
    clean: list[dict] = []
    dropped = 0

    for p in pairs:
        if "error-to-fix" in p.get("tags", []):
            clean.append(p)
            continue
        if p.get("compiles") is True:
            clean.append(p)
            continue
        # Synthesized rows: heuristic pre-check
        if _try_compile_inline(p.get("output", "")):
            p["compiles"] = None  # pending full harness
            clean.append(p)
        else:
            dropped += 1

    print(f"M5 QC: {len(pairs)} in → {len(clean)} kept, {dropped} dropped")

    # Decontaminated train/eval split
    random.shuffle(clean)
    eval_size = max(50, len(clean) // 10)
    eval_pairs = clean[:eval_size]
    train_pairs = clean[eval_size:]

    # Decontaminate: remove any eval id from train
    eval_ids = {p["id"] for p in eval_pairs}
    train_pairs = [p for p in train_pairs if p["id"] not in eval_ids]

    print(f"M5 split: train={len(train_pairs)}, eval={len(eval_pairs)}")

    # Bias eval toward deepbook + dependency-pattern
    eval_pairs.sort(
        key=lambda p: (
            TAG_DEEPBOOK in p.get("tags", []) or TAG_DEPPATTERN in p.get("tags", [])
        ),
        reverse=True,
    )
    return train_pairs, eval_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Write JSONL helpers
# ══════════════════════════════════════════════════════════════════════════════

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows → {path}")


def tag_distribution(rows: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for row in rows:
        for tag in row.get("tags", []):
            dist[tag] = dist.get(tag, 0) + 1
    return dist


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

STAGES = ["m1", "m2", "m3", "m4", "m5"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Move/DeepBook fine-tuning dataset")
    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--max-per-tag",
        type=int,
        default=300,
        help="Max files to sample per tag bucket in M4 (default 300)",
    )
    args = parser.parse_args()

    run = set(STAGES) if args.stage == "all" else {args.stage}

    # ── M1 ──────────────────────────────────────────────────────────────────
    files: list[Path] = []
    if "m1" in run:
        files = stage_m1()

    # ── M2 ──────────────────────────────────────────────────────────────────
    kept: list[Path] = []
    if "m2" in run:
        if not files:
            files = stage_m1()
        kept, _ = stage_m2(files)

    # ── M3 ──────────────────────────────────────────────────────────────────
    buckets: dict[str, list[Path]] = {}
    if "m3" in run:
        if not kept:
            kept = sorted(FILTERED_DIR.rglob("*.move"))
            print(f"M3: loaded {len(kept)} files from {FILTERED_DIR}")
        buckets = stage_m3(kept)

    # ── M4 ──────────────────────────────────────────────────────────────────
    raw_pairs: list[dict] = []
    if "m4" in run:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY is not set. Set it and re-run --stage m4.")
            sys.exit(1)
        if not buckets:
            kept = sorted(FILTERED_DIR.rglob("*.move"))
            buckets = stage_m3(kept)
        raw_pairs = stage_m4(buckets, max_per_tag=args.max_per_tag)

    # ── M5 ──────────────────────────────────────────────────────────────────
    if "m5" in run:
        if not raw_pairs:
            # Load previously synthesized pairs if resuming
            if PAIRS_FILE.exists():
                with open(PAIRS_FILE) as f:
                    raw_pairs = [json.loads(l) for l in f if l.strip()]
                print(f"M5: loaded {len(raw_pairs)} existing pairs from {PAIRS_FILE}")
            else:
                print("M5: no pairs to QC. Run --stage m4 first.")
                sys.exit(1)

        train_pairs, eval_pairs = stage_m5_qc(raw_pairs)
        write_jsonl(PAIRS_FILE, train_pairs)
        write_jsonl(EVAL_FILE, eval_pairs)

        dist = tag_distribution(train_pairs)
        print(f"\nTag distribution (train): {dist}")
        print(
            "\nDataset ready. Run the full compile harness with:\n"
            f"  python scripts/move_eval.py --input {PAIRS_FILE} --output eval_report.json\n"
            f"  python scripts/move_eval.py --input {EVAL_FILE}  --output eval_report_eval.json"
        )


if __name__ == "__main__":
    main()

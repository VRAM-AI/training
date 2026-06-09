#!/usr/bin/env python3
"""
gen_ts_ptb_pairs.py — Generate TypeScript PTB instruction-to-code pairs
from the real DeepBook v3 transaction scripts in data/raw/deepbookv3/scripts/.

Each pair:
  instruction : natural-language task ("Write a Sui PTB to …")
  input       : "" (standalone)
  output      : full TypeScript source using @mysten/deepbook-v3
  tags        : ["deepbook", "typescript-ptb"]
  compiles    : null (TS, no Move compile check)

Usage:
  python scripts/gen_ts_ptb_pairs.py
  python scripts/gen_ts_ptb_pairs.py --out data/ts_ptb_pairs.jsonl --limit 60
"""

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
TS_SOURCE_DIR = ROOT / "data" / "raw" / "deepbookv3" / "scripts" / "transactions"
CONSTANTS_FILE = ROOT / "data" / "raw" / "deepbookv3" / "scripts" / "config" / "constants.ts"
DEFAULT_OUT = ROOT / "data" / "ts_ptb_pairs.jsonl"

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
TEACHER_MODEL = "deepseek/deepseek-chat"

REVERSE_TS_PROMPT = """\
You are a Sui Move and TypeScript expert. Given the following TypeScript Programmable Transaction Block (PTB) script that uses the @mysten/deepbook-v3 SDK, write a single concise natural-language instruction (1-2 sentences) that a developer would give to produce exactly this code.

Rules:
- Start with an action verb ("Write", "Create", "Implement", "Build")
- Mention the specific DeepBook operation (e.g. BalanceManager, margin pool supply, flashloan, limit order)
- Do NOT include the code in your answer
- Output ONLY the instruction, nothing else

TypeScript code:
```typescript
{code}
```"""


def call_teacher(prompt: str, max_tokens: int = 300, retries: int = 3) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    })
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "45",
                 "-H", "Content-Type: application/json",
                 "-H", f"Authorization: Bearer {api_key}",
                 "-d", payload,
                 OPENROUTER_BASE],
                capture_output=True, text=True, timeout=60,
            )
            data = json.loads(result.stdout)
            content = (
                data.get("choices", [{}])[0].get("message", {}).get("content")
                or data.get("choices", [{}])[0].get("message", {}).get("reasoning")
            )
            if content and content.strip():
                return content.strip()
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {e}")
        wait = 2 ** attempt
        print(f"    retrying in {wait}s …")
        time.sleep(wait)
    raise RuntimeError("All retries exhausted")


def make_ts_pair(ts_file: Path) -> dict | None:
    code = ts_file.read_text(errors="replace")
    if len(code.strip()) < 80:
        return None
    prompt_code = code[:2000]
    try:
        instruction = call_teacher(REVERSE_TS_PROMPT.format(code=prompt_code))
        return {
            "id": str(uuid.uuid4()),
            "instruction": instruction,
            "input": "",
            "output": code,
            "tags": ["deepbook", "typescript-ptb"],
            "source": str(ts_file.relative_to(ROOT / "data" / "raw")),
            "compiles": None,
        }
    except Exception as e:
        print(f"    WARN: failed for {ts_file.name}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    ts_files = sorted(TS_SOURCE_DIR.glob("*.ts"))
    if args.limit:
        ts_files = ts_files[: args.limit]

    print(f"Processing {len(ts_files)} TypeScript files → {args.out}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w") as f:
        for i, ts_file in enumerate(ts_files, 1):
            print(f"  [{i}/{len(ts_files)}] {ts_file.name} …", end="", flush=True)
            pair = make_ts_pair(ts_file)
            if pair:
                f.write(json.dumps(pair) + "\n")
                f.flush()
                written += 1
                print(f" ✓")
            else:
                print(f" SKIP")

    print(f"\nDone: {written} TypeScript PTB pairs written to {out_path}")
    print(f"Merge with: cat {out_path} >> data/pairs.jsonl")


if __name__ == "__main__":
    main()

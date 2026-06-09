#!/usr/bin/env python3
"""
gen_margin_predict_pairs.py — Generate targeted Move instruction pairs for
DeepBook Margin (borrow/lend/leverage) and Predict (binary markets/options)
modules from data/raw/deepbookv3/packages/deepbook_margin and /predict.

For each source file we generate up to 2 pairs:
  1. reverse-instruction  (what does this module do?)
  2. scenario-specific    (targeted prompt based on keywords found in the file)

Tags:
  margin files  → ["deepbook", "margin"]
  predict files → ["deepbook", "predict"]

Usage:
  python scripts/gen_margin_predict_pairs.py
  python scripts/gen_margin_predict_pairs.py --out data/margin_predict_pairs.jsonl
"""

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
PACKAGES_DIR = ROOT / "data" / "raw" / "deepbookv3" / "packages"
DEFAULT_OUT = ROOT / "data" / "margin_predict_pairs.jsonl"

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
TEACHER_MODEL = "deepseek/deepseek-chat"

REVERSE_PROMPT = """\
You are a Sui Move expert. Given the Move module below, write a single concise natural-language instruction (1-2 sentences) that a developer would type to produce exactly this code.

Rules:
- Start with an action verb ("Implement", "Write", "Create", "Define")
- Name the specific DeepBook primitive (margin pool, leverage, borrow, repay, liquidation, binary prediction pool, options pricing, Black-Scholes, etc.)
- Do NOT include the code in your answer
- Output ONLY the instruction, nothing else

Move code:
```move
{code}
```"""

SCENARIO_PROMPTS = {
    "borrow": "Write a Move function that allows a user to borrow {asset} from a DeepBook margin pool, checking the pool's available liquidity and recording the debt position.",
    "repay": "Write a Move function that repays a DeepBook margin loan, burning the debt record and releasing the collateral back to the user.",
    "leverage": "Write a Move module that opens a leveraged long position on DeepBook using a BalanceManager, supporting up to {leverage}x leverage with proper health-factor checks.",
    "liquidation": "Implement a DeepBook margin liquidation function that checks whether a position is undercollateralised and liquidates it, distributing proceeds to the liquidator and protocol.",
    "supply": "Write a Move function that supplies {asset} liquidity to a DeepBook margin pool, minting LP shares proportional to the deposit.",
    "binary_pool": "Create a Move module that deploys a DeepBook Predict binary prediction market pool with configurable strike price, expiry, and oracle settlement.",
    "options_pricing": "Implement a Move function that calculates the Black-Scholes price for a DeepBook Predict options contract given spot price, strike, volatility, time-to-expiry, and risk-free rate.",
    "oracle": "Write a Move module that integrates an on-chain oracle price feed into a DeepBook Predict market for settlement at expiry.",
    "strike_exposure": "Implement a Move function that calculates and caps the maximum strike exposure for a DeepBook Predict position to prevent pool insolvency.",
    "ewma": "Write a Move module that computes an exponentially weighted moving average (EWMA) of the underlying asset price for DeepBook Predict volatility estimation.",
}

KEYWORD_TO_SCENARIO = {
    "borrow": "borrow",
    "repay": "repay",
    "leverage": "leverage",
    "liquidat": "liquidation",
    "supply": "supply",
    "binary": "binary_pool",
    "black_scholes": "options_pricing",
    "option": "options_pricing",
    "oracle": "oracle",
    "strike_exposure": "strike_exposure",
    "ewma": "ewma",
}


def call_teacher(prompt: str, max_tokens: int = 600, retries: int = 3) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.5,
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
            print(f"    attempt {attempt+1} error: {e}")
        wait = 2 ** attempt
        print(f"    retry in {wait}s …")
        time.sleep(wait)
    raise RuntimeError("All retries exhausted")


def detect_scenario(code: str) -> str | None:
    code_lower = code.lower()
    for kw, scenario in KEYWORD_TO_SCENARIO.items():
        if kw in code_lower:
            return scenario
    return None


def make_pairs_for_file(f: Path, tags: list[str]) -> list[dict]:
    full_code = f.read_text(errors="replace")
    if len(full_code.strip()) < 100:
        return []
    prompt_code = full_code[:2000]
    source = str(f.relative_to(ROOT / "data" / "raw"))
    results = []

    # Pair 1: reverse-instruction
    try:
        instruction = call_teacher(REVERSE_PROMPT.format(code=prompt_code))
        results.append({
            "id": str(uuid.uuid4()),
            "instruction": instruction,
            "input": "",
            "output": full_code,
            "tags": tags,
            "source": source,
            "compiles": True,
        })
    except Exception as e:
        print(f"    WARN reverse failed for {f.name}: {e}")

    # Pair 2: scenario-specific synthesis
    scenario = detect_scenario(full_code)
    if scenario:
        template = SCENARIO_PROMPTS[scenario]
        # Fill template placeholders with reasonable defaults
        scenario_prompt = template.format(asset="USDC", leverage="3")
        synthesis_prompt = f"""\
You are a Sui Move 2024 expert writing for the DeepBook v3 protocol. Write a complete, compilable Move module that satisfies the following requirement. Output ONLY raw Move code with no markdown fences, no explanations.

Requirement: {scenario_prompt}

Use Move 2024 edition syntax (method/receiver style where applicable). Reference real DeepBook module paths where needed."""
        try:
            output_code = call_teacher(synthesis_prompt, max_tokens=1200)
            if "module" in output_code and "fun " in output_code:
                results.append({
                    "id": str(uuid.uuid4()),
                    "instruction": scenario_prompt,
                    "input": "",
                    "output": output_code,
                    "tags": tags + ["synthesized"],
                    "source": f"synthesized/{scenario}",
                    "compiles": None,
                })
        except Exception as e:
            print(f"    WARN scenario failed for {f.name}/{scenario}: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    # Collect source files
    jobs: list[tuple[Path, list[str]]] = []
    for f in sorted((PACKAGES_DIR / "deepbook_margin").rglob("*.move")):
        jobs.append((f, ["deepbook", "margin"]))
    for f in sorted((PACKAGES_DIR / "predict").rglob("*.move")):
        jobs.append((f, ["deepbook", "predict"]))
    for f in sorted((PACKAGES_DIR / "margin_liquidation").rglob("*.move")):
        jobs.append((f, ["deepbook", "margin"]))

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Processing {len(jobs)} files → {args.out}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w") as fout:
        for i, (f, tags) in enumerate(jobs, 1):
            pillar = "margin" if "margin" in str(f) else "predict"
            print(f"  [{i}/{len(jobs)}] [{pillar}] {f.name} …", end="", flush=True)
            pairs = make_pairs_for_file(f, tags)
            for p in pairs:
                fout.write(json.dumps(p) + "\n")
                fout.flush()
            total += len(pairs)
            print(f" → {len(pairs)} pair(s)")

    print(f"\nDone: {total} margin/predict pairs → {out_path}")
    print(f"Merge with: cat {out_path} >> data/pairs.jsonl")


if __name__ == "__main__":
    main()

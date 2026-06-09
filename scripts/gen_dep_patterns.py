#!/usr/bin/env python3
"""
gen_dep_patterns.py  –  Generate dependency-pattern pairs.

These pairs cannot come from the corpus (cyclic-dep code doesn't compile by
definition). We use the teacher model to synthesize broken→fixed pairs that
demonstrate the five escape patterns Tony Lee flagged:

  1. Witness pattern
  2. Dynamic fields (no hard dependency on concrete type)
  3. Splitting shared types into a base module
  4. public(package) visibility
  5. Generic dependency inversion

Each pair:
  instruction : "Fix the circular dependency in the following Move code."
  input       : the cyclic (broken) version
  output      : the fixed version
  tags        : ["dependency-pattern", "error-to-fix"]
  compiles    : null  (fixed code verified by move_eval.py separately)

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/gen_dep_patterns.py --count 50 --out data/dep_patterns.jsonl
    # Then merge into pairs.jsonl:
    cat data/dep_patterns.jsonl >> data/pairs.jsonl
"""

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
TEACHER_MODEL = "deepseek/deepseek-chat"  # DeepSeek V3 — fast, non-reasoning

PATTERNS = [
    {
        "name": "witness",
        "description": "two modules with a circular import fixed by the Witness pattern (a one-time proof struct passed across the boundary)",
    },
    {
        "name": "dynamic-fields",
        "description": "a module that stores a concrete type from another module (causing a cycle) fixed by replacing the field with a dynamic field keyed by a marker struct",
    },
    {
        "name": "base-module-split",
        "description": "two modules that each import the other's shared struct fixed by extracting the shared struct into a third base module that both import",
    },
    {
        "name": "public-package-visibility",
        "description": "a module exposing an internal helper as `public` (causing a dependent to import it in a cycle) fixed by changing the helper to `public(package)` visibility",
    },
    {
        "name": "generic-inversion",
        "description": "a module parameterized on a concrete type from another module (cycle) fixed by making the dependency generic and passing the type at call site",
    },
]

PROMPT_TEMPLATE = """\
You are an expert in Sui Move 2024 (edition "2024.beta") and Move's acyclic dependency model.

Generate a circular-dependency example and its fix for this pattern:
  Pattern: {name}
  Description: {description}

Respond in this EXACT format (no extra text, no markdown fences):

INSTRUCTION: Fix the circular dependency in the following Move code using the {name} pattern.

BROKEN:
<complete broken Move code – must have a real circular import that `sui move build` would reject>

ERROR: <one-line compiler error the developer would see>

FIXED:
<complete fixed Move code – compilable, uses 2024 method syntax where natural, adds a one-line comment explaining WHY the fix works>
"""


def call_teacher(prompt: str, retries: int = 3) -> str:
    """Call OpenRouter via curl subprocess (avoids macOS Python DNS issues)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")

    for attempt in range(1, retries + 1):
        payload = json.dumps({
            "model": TEACHER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2500,
            "temperature": 0.4,
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
                    "-H", "X-Title: Move Dep-Pattern Generator",
                    "-d", payload,
                ],
                capture_output=True, text=True, timeout=60,
            )
            data = json.loads(result.stdout)
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning") or ""
            if content:
                return content.strip()
            print(f"  WARN empty response (attempt {attempt}/{retries})")
        except Exception as e:
            print(f"  WARN attempt {attempt}/{retries} failed: {e}")
        time.sleep(2 * attempt)
    return ""


def parse_response(text: str) -> dict | None:
    """Parse the structured teacher response into fields."""
    try:
        instruction = ""
        broken = ""
        error = ""
        fixed = ""

        if "INSTRUCTION:" in text:
            instruction = text.split("INSTRUCTION:", 1)[1].split("BROKEN:")[0].strip()
        if "BROKEN:" in text:
            broken = text.split("BROKEN:", 1)[1].split("ERROR:")[0].strip()
        if "ERROR:" in text:
            error = text.split("ERROR:", 1)[1].split("FIXED:")[0].strip()
        if "FIXED:" in text:
            fixed = text.split("FIXED:", 1)[1].strip()

        if not (instruction and broken and fixed):
            return None

        return {
            "id": str(uuid.uuid4()),
            "instruction": instruction,
            "input": f"// Compiler error: {error}\n\n{broken}",
            "output": fixed,
            "tags": ["dependency-pattern", "error-to-fix"],
            "source": "synthesized/dep-pattern",
            "compiles": None,
        }
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dependency-pattern pairs")
    parser.add_argument("--count", type=int, default=50, help="Pairs to generate per pattern")
    parser.add_argument("--out", default="data/dep_patterns.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w") as f:
        for pattern in PATTERNS:
            target = args.count
            generated = 0
            attempts = 0
            print(f"Generating {target} pairs for pattern: {pattern['name']}")
            while generated < target and attempts < target * 3:
                attempts += 1
                try:
                    response = call_teacher(
                        PROMPT_TEMPLATE.format(
                            name=pattern["name"],
                            description=pattern["description"],
                        )
                    )
                    pair = parse_response(response)
                    if pair:
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        f.flush()
                        generated += 1
                        total += 1
                        print(f"  [{generated}/{target}] {pair['id']}")
                    else:
                        print(f"  attempt {attempts}: parse failed, retrying")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  attempt {attempts} error: {e}")
                    time.sleep(2)

    print(f"\nDone. Generated {total} dependency-pattern pairs → {out_path}")
    print(f"Merge into training set with:  cat {out_path} >> data/pairs.jsonl")


if __name__ == "__main__":
    main()

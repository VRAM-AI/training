#!/usr/bin/env python3
"""
move_eval.py  –  Compile-based eval harness for the Move/DeepBook dataset.

Usage:
    python scripts/move_eval.py --input data/pairs.jsonl --output eval_report.json
    python scripts/move_eval.py --input data/eval.jsonl  --output eval_report.json --model my-model

What it does for each row:
  1. Drops the `output` code into a scaffolded Sui Move package.
  2. Runs `sui move build` and captures pass/fail + compiler error.
  3. Checks for Move 2024 method/receiver syntax usage.
  4. Checks DeepBook API presence for deepbook-tagged rows.
  5. Emits eval_report.json in the Section 7.3 schema.

Requirements:
  - `sui` CLI on PATH  (https://docs.sui.io/references/cli)
  - Python 3.9+
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

# ── Repo-local paths (resolved relative to this script) ──────────────────────
_SCRIPTS_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_DEEPBOOK_TOKEN = _REPO_ROOT / "data" / "raw" / "deepbookv3" / "packages" / "token"
_DEEPBOOK_PKG = _REPO_ROOT / "data" / "raw" / "deepbookv3" / "packages" / "deepbook"

# NOTE: deepbookv3 main branch requires a newer Sui framework version than what
# ships with the local sui CLI (1.52.3). Compiling deepbook-tagged rows against
# the local deepbook package fails with upstream incompatibilities (e.g.
# vec_set.length). Deepbook-tagged synthesized rows are therefore evaluated
# with semantic checks only (API presence + 2024 syntax). Set this env var to
# '1' once the CLI is upgraded to re-enable full compilation for deepbook rows.
_DEEPBOOK_COMPILE_ENABLED = os.environ.get("MOVE_EVAL_DEEPBOOK_COMPILE", "0") == "1"

# ── DeepBook v3 canonical API surface ────────────────────────────────────────
DEEPBOOK_APIS = [
    "balance_manager",
    "BalanceManager",
    "TradeProof",
    "trade_proof",
    "place_limit_order",
    "place_market_order",
    "flashloan",
    "flash_loan",
    "DEEP",
    "deepbook",
]

# ── Move 2024 method-syntax indicator: `receiver.method(` call pattern ────────
METHOD_SYNTAX_RE = re.compile(r"\w+\.\w+\s*\(")

# Addresses that belong to framework packages – can't be reused in a user package
_RESERVED_ADDRS = {"sui", "std", "bridge", "sui_system", "deepbook", "token", "move_stdlib"}


def _extract_module_address(code: str) -> str:
    """Extract the package address from code; return 'eval_pkg' for reserved/unknown addrs."""
    m = re.search(r"\bmodule\s+(\w+)::", code)
    if not m:
        return "eval_pkg"
    addr = m.group(1)
    return "eval_pkg" if addr in _RESERVED_ADDRS else addr


def _normalize_module_address(code: str, pkg_addr: str) -> str:
    """Rewrite `module <old>::name` to `module <pkg_addr>::name` if old was reserved."""
    m = re.search(r"\bmodule\s+(\w+)::", code)
    if not m:
        return code
    old_addr = m.group(1)
    if old_addr in _RESERVED_ADDRS:
        return re.sub(rf"\bmodule\s+{old_addr}::", f"module {pkg_addr}::", code, count=1)
    return code


def _make_scaffold_toml(
    pkg_addr: str = "eval_pkg",
    deepbook_path: Path | None = None,
) -> str:
    """Generate Move.toml. Only declare deepbook as direct dep (it transitively pulls token)."""
    dep_lines = []
    if deepbook_path:
        dep_lines.append(f'deepbook = {{ local = "{deepbook_path}" }}')
    deps_block = "\n".join(dep_lines) if dep_lines else ""

    # Always use 'eval_pkg' as package name to avoid collision with dep names.
    addr_lines = ['eval_pkg = "0x0"']
    if deepbook_path:
        addr_lines.append('deepbook = "0x0"')
        addr_lines.append('token = "0x0"')
    # If the code uses a different address name, add it too
    if pkg_addr not in ("eval_pkg", "deepbook", "token"):
        addr_lines.append(f'{pkg_addr} = "0x0"')
    addrs_block = "\n".join(addr_lines)

    return f"""[package]
name = "eval_pkg"
edition = "2024.beta"

[dependencies]
{deps_block}

[addresses]
{addrs_block}
"""


def _patch_deepbook_toml(tmpdir: str) -> tuple[Path, Path]:
    """
    Copy token + deepbook packages into tmpdir and rewrite deepbook's Move.toml
    so it points at the local token copy instead of git. Returns (token_path, deepbook_path).
    Idempotent: safe to call multiple times with the same tmpdir.
    """
    token_dest = Path(tmpdir) / "_deps" / "token"
    db_dest = Path(tmpdir) / "_deps" / "deepbook"
    if token_dest.exists() and db_dest.exists():
        return token_dest, db_dest
    shutil.copytree(str(_DEEPBOOK_TOKEN), str(token_dest))
    shutil.copytree(str(_DEEPBOOK_PKG), str(db_dest))

    # Patch deepbook/Move.toml: replace git token dep with local path
    toml_path = db_dest / "Move.toml"
    toml_text = toml_path.read_text()
    toml_text = re.sub(
        r'token\s*=\s*\{[^}]+\}',
        f'token = {{ local = "{token_dest}" }}',
        toml_text,
    )
    toml_path.write_text(toml_text)
    return token_dest, db_dest


def scaffold_package(code: str, tmpdir: str, needs_deepbook: bool = False) -> Path:
    """Write a minimal Sui package around the candidate code and return pkg dir."""
    pkg_addr = _extract_module_address(code)
    pkg = Path(tmpdir) / pkg_addr
    sources = pkg / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    if needs_deepbook and _DEEPBOOK_PKG.exists():
        _token_dest, db_dest = _patch_deepbook_toml(tmpdir)
        toml = _make_scaffold_toml(pkg_addr, deepbook_path=db_dest)
    else:
        toml = _make_scaffold_toml(pkg_addr)

    (pkg / "Move.toml").write_text(toml)

    # Normalize reserved addresses so the code builds under our eval_pkg address
    code = _normalize_module_address(code, pkg_addr)

    # If the code already contains a full module declaration, use it directly.
    if re.search(r"\bmodule\s+\w", code):
        (sources / "eval_module.move").write_text(code)
    else:
        # Wrap bare functions / structs in a module.
        body = textwrap.indent(code, "    ")
        wrapped = f"module {pkg_addr}::eval_module {{\n{body}\n}}"
        (sources / "eval_module.move").write_text(wrapped)

    return pkg


def run_sui_build(pkg_dir: Path, timeout: int = 60) -> tuple[bool, str]:
    """Run `sui move build` and return (success, stderr_or_empty)."""
    try:
        result = subprocess.run(
            ["sui", "move", "build", "--skip-fetch-latest-git-deps"],
            cwd=str(pkg_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout).strip()
    except FileNotFoundError:
        return False, "sui CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return False, f"build timed out after {timeout}s"


def uses_method_syntax(code: str) -> bool:
    return bool(METHOD_SYNTAX_RE.search(code))


def has_deepbook_api(code: str) -> bool:
    return any(api in code for api in DEEPBOOK_APIS)


def eval_row(row: dict[str, Any], tmpdir: str) -> dict[str, Any]:
    code: str = row.get("output", "")
    tags: list[str] = row.get("tags", [])
    is_error_fix: bool = "error-to-fix" in tags
    is_deepbook: bool = "deepbook" in tags

    result: dict[str, Any] = {
        "id": row.get("id", ""),
        "tags": tags,
        "compiles": False,
        "compiler_error": "",
        "uses_2024_syntax": uses_method_syntax(code),
        "deepbook_api_present": has_deepbook_api(code) if is_deepbook else None,
        "skipped_compile": is_error_fix,
    }

    if is_error_fix:
        # error-to-fix rows intentionally contain broken code; skip compile.
        result["compiles"] = None
        return result

    if is_deepbook and not _DEEPBOOK_COMPILE_ENABLED:
        # Upstream deepbook package incompatible with local CLI; semantic-check only.
        result["compiles"] = None
        result["compiler_error"] = "deepbook compile skipped (set MOVE_EVAL_DEEPBOOK_COMPILE=1 to enable)"
        return result

    pkg = scaffold_package(code, tmpdir, needs_deepbook=is_deepbook)
    ok, err = run_sui_build(pkg)
    result["compiles"] = ok
    result["compiler_error"] = err if not ok else ""
    # clean up so each row gets a fresh package
    shutil.rmtree(str(pkg), ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Move compile-rate eval harness")
    parser.add_argument("--input", required=True, help="Path to .jsonl file to evaluate")
    parser.add_argument("--output", default="eval_report.json", help="Output report path")
    parser.add_argument("--model", default="reference", help="Model name tag for the report")
    parser.add_argument("--timeout", type=int, default=60, help="Per-row build timeout (s)")
    parser.add_argument(
        "--row-results", default="", help="Optional path to write per-row JSONL results"
    )
    args = parser.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No rows found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(rows)} rows from {args.input} …")

    row_results = []
    with tempfile.TemporaryDirectory(prefix="move_eval_") as tmpdir:
        for i, row in enumerate(rows):
            res = eval_row(row, tmpdir)
            row_results.append(res)
            status = (
                "SKIP(error-fix)"
                if res["skipped_compile"]
                else ("OK" if res["compiles"] else "FAIL")
            )
            print(f"  [{i+1}/{len(rows)}] {res['id']}: {status}")

    # ── Aggregate stats ──────────────────────────────────────────────────────
    compile_eligible = [r for r in row_results if r["compiles"] is not None]
    n_compile = len(compile_eligible)
    n_pass = sum(1 for r in compile_eligible if r["compiles"])
    compile_rate = (n_pass / n_compile) if n_compile else 0.0

    syntax_rows = [r for r in row_results if r["uses_2024_syntax"] is not None]
    syntax_rate = (
        sum(1 for r in syntax_rows if r["uses_2024_syntax"]) / len(syntax_rows)
        if syntax_rows
        else 0.0
    )

    deepbook_rows = [r for r in row_results if r["deepbook_api_present"] is not None]
    db_pass_rate = (
        sum(1 for r in deepbook_rows if r["deepbook_api_present"] and r["compiles"])
        / len(deepbook_rows)
        if deepbook_rows
        else 0.0
    )

    report = {
        "model": args.model,
        "n": len(rows),
        "compile_rate": round(compile_rate, 4),
        "uses_2024_syntax_rate": round(syntax_rate, 4),
        "deepbook_task_pass_rate": round(db_pass_rate, 4),
        "timestamp_ms": int(time.time() * 1000),
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    if args.row_results:
        with open(args.row_results, "w") as f:
            for r in row_results:
                f.write(json.dumps(r) + "\n")

    print(f"\n── Eval report ──────────────────────────────")
    print(f"  Rows evaluated  : {len(rows)}")
    print(f"  Compile-eligible: {n_compile}")
    print(f"  Compile rate    : {compile_rate:.1%}")
    print(f"  Method-syntax   : {syntax_rate:.1%}")
    print(f"  DeepBook pass   : {db_pass_rate:.1%}")
    print(f"  Report written  : {args.output}")


if __name__ == "__main__":
    main()

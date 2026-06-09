# Move / DeepBook Fine-Tuning Dataset

Produces a high-quality instruction-to-code dataset for fine-tuning a small open model into a **Sui Move 2024 + DeepBook v3** coding specialist.

Target model: **Gemma 4 E2B** (`google/gemma-4-E2B-it`) trained on **vram.ai** with QLoRA.  
Teacher model: **DeepSeek V4 Flash** via OpenRouter.

---

## Quick start

```bash
# 1. Install Python deps (stdlib only – no third-party packages required for pipeline)
python --version   # 3.9+

# 2. Set your OpenRouter key (only needed for M4 synthesis)
cp .env.example .env
# edit .env and fill in OPENROUTER_API_KEY

# 3. Run M1 → M2 → M3 (no API key needed)
python scripts/build_move_dataset.py --stage m1
python scripts/build_move_dataset.py --stage m2
python scripts/build_move_dataset.py --stage m3

# 4. Synthesize pairs (requires OPENROUTER_API_KEY)
source .env   # or export OPENROUTER_API_KEY=...
python scripts/build_move_dataset.py --stage m4

# 5. QC + train/eval split
python scripts/build_move_dataset.py --stage m5

# 6. Run the full compile harness (requires `sui` CLI on PATH)
python scripts/move_eval.py --input data/pairs.jsonl --output eval_report.json
python scripts/move_eval.py --input data/eval.jsonl  --output eval_report_eval.json
```

---

## Deliverables

| File | Description |
|---|---|
| `data/pairs.jsonl` | Training pairs (Section 7.1 schema) |
| `data/eval.jsonl` | Hold-out eval pairs (decontaminated) |
| `data/SOURCES.md` | All data sources with URLs + licenses |
| `data/filter_report.json` | M2 filter count report |
| `scripts/build_move_dataset.py` | Full pipeline |
| `scripts/move_eval.py` | Compile-based eval harness |
| `training/MODELS.md` | HF repo IDs + teacher model config |

---

## Pair schema (data/pairs.jsonl)

```json
{
  "id": "string",
  "instruction": "natural-language task",
  "input": "optional context or partial code",
  "output": "the Move code answer",
  "tags": ["deepbook" | "receiver-syntax" | "dependency-pattern" | "general"],
  "source": "provenance string",
  "compiles": true
}
```

---

## Eval report schema (eval_report.json)

```json
{
  "model": "string",
  "n": 0,
  "compile_rate": 0.0,
  "uses_2024_syntax_rate": 0.0,
  "deepbook_task_pass_rate": 0.0,
  "timestamp_ms": 0
}
```

---

## Guardrails

- No secrets committed. Scan before every push.  
- Only permissive licenses (MIT / Apache 2.0). See `data/SOURCES.md`.  
- Non-compiling synthesized rows are dropped in M5, unless tagged `error-to-fix`.  
- Dataset is **private** (internal use only).

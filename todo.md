# Work Order: Move/DeepBook Fine-Tuning Dataset

**Owner:** Hermes agent, executing inside the `VRAM-HUB` repo.
**Goal:** produce a high-quality instruction-to-code dataset that turns a small open model into a Sui Move + DeepBook coding specialist, plus the compile-based eval harness that gates quality.
**Origin:** requirements from a Mysten/DeepBook engineer (Tony Lee). The model must be current on Move 2024, use method (receiver) syntax, respect Move's acyclic dependency model, and know DeepBook v3 integration.

Work the milestones in order. Each task has an acceptance criterion. Do not mark a task done until its criterion passes. Where a decision is the human's, it is listed in Section 8. Ask, do not guess.

---

## 0. Guardrails (read first, non-negotiable)

- Keep `VRAM-HUB` private. Do not change repo visibility. Dataset tooling lives here; the public face is the future `vram-sdk`.
- Never commit secrets. No mnemonics, R2 keys, or `.env` values in any file or dataset row. Scan every file before commit.
- License discipline. Record the license of every data source in `data/SOURCES.md`. Use only permissive sources (MIT / Apache / public docs).
- No fabricated data passed off as real. A synthesized pair whose output does not compile is dropped, unless it is a deliberate error-to-fix example that is tagged as such.
- The dataset is model-agnostic JSONL. Building it does not depend on the final base-model choice, so proceed now.

---

## 1. Locked decisions

| Item | Choice | Note |
|---|---|---|
| Target model (fine-tune + deploy) | Gemma 4 E2B (Apache 2.0, browser-deployable) | Final confirm in Section 8. Fallback: a small Qwen coder if compile-rate is poor. Does not block data prep. |
| Teacher model (instruction synthesis) | DeepSeek V4 (MIT) | Top coding model; MIT output is redistributable so the dataset can be published. Use the Flash tier via API unless the human says otherwise. |
| Method | QLoRA | Context only; not part of this work order. |
| Deliverables | `data/pairs.jsonl`, `data/eval.jsonl`, `data/SOURCES.md`, `scripts/build_move_dataset.py`, `scripts/move_eval.py` | |
| Target size | A few thousand to low tens of thousands of high-quality pairs | Quality over volume. |

**TODO-VERIFY (do not guess):** confirm the exact Hugging Face repo ids for Gemma 4 E2B and the DeepSeek V4 teacher tier before writing any loader or API code. Record them in `training/MODELS.md`.

---

## 2. M1 - Collect the raw corpus

Clone and extract Move source from:

- Sui framework: the `move-stdlib` and `sui-framework` packages in `MystenLabs/sui`. The most idiomatic, current Move that exists.
- DeepBook v3: `MystenLabs/deepbookv3`, including its `tests/` and any examples. This is the DeepBook ground truth.
- Sui example packages and the Sui docs (docs pages already pair prose with code snippets, which are near-ready instruction/code pairs).
- Selected third-party Sui packages on GitHub, permissive license only.

**Done when:** `data/raw/` contains the source trees and `data/SOURCES.md` lists every source with its URL and license.

---

## 3. M2 - Filter to current and correct ("all the Move", done right)

- Keep only the 2024 edition. Check the `edition` field in each `Move.toml`; drop pre-2024 packages.
- Compile everything with `sui move build`. Keep only packages/modules that compile. Compiling code is the floor.
- Deduplicate near-identical files so the model does not memorize and so the eval set stays clean.

**Done when:** `data/filtered/` contains only 2024-edition, compiling, deduplicated code, with a count report (files in, files kept, reasons dropped).

---

## 4. M3 - Build four targeted subsets (one per Tony requirement)

Curate these deliberately. Do not assume the broad corpus covers them.

1. **Method / receiver syntax.** Collect functions that take `self` as the first parameter and are called with dot notation. Add "rewrite to method syntax" pairs to teach the idiom directly.
   - *Done when:* at least 200 curated, compiling examples, tagged `receiver-syntax`.
2. **Circular-dependency avoidance.** The pattern a naive model gets wrong. Curate broken-to-fixed pairs that show the escape patterns: witness, dynamic fields, splitting shared types into a base module, `public(package)` visibility, generic dependency inversion. Each pair: the cyclic version, the fix, and a one-line why. Mostly hand-curated plus synthesized.
   - *Done when:* at least 200 examples tagged `dependency-pattern`, fixes compile.
3. **DeepBook v3 integration.** From the `deepbookv3` tests and examples: BalanceManager, TradeProof, placing limit and market orders, flashloans, the DEEP fee model. Turn each into a "do X with DeepBook" to correct-code pair.
   - *Done when:* at least 200 examples tagged `deepbook`, all compile.
4. **General Move correctness.** The broad compiled corpus from M2.

---

## 5. M4 - Synthesize instruction-to-code pairs

Raw code is not enough for an assistant. Generate (instruction, optional input, output) triples using the teacher model (DeepSeek V4) and mining:

- **Reverse-instruction (workhorse):** take a real function, have the teacher write the natural-language task that produces it.
- **Doc-to-code:** the Sui docs snippets are already paired; mine directly.
- **Commit-to-diff:** mine git history; commit message as instruction, diff as the change. Good for "modify this" tasks.
- **Error-to-fix:** take compiling code, inject a realistic error (especially a circular-dependency or wrong-syntax one), run the compiler to capture the real error message, and pair (broken code + error -> fix). Directly trains the behaviors Tony flagged.

Tag every pair (`deepbook` / `receiver-syntax` / `dependency-pattern` / `general`).

**Done when:** `data/pairs.jsonl` exists in the Section 7 schema, deduplicated, with a tag-distribution report.

---

## 6. M5 - QC, decontaminate, and the eval harness

- **Eval harness first.** `scripts/move_eval.py` takes generated Move, drops it into a scaffolded package, runs `sui move build`, and records pass/fail plus the compiler error. Add checks for: uses 2024 method syntax (yes/no), no circular deps (compiling implies this), DeepBook calls present and correct for DeepBook tasks. Emits `eval_report.json` (schema in Section 7.3).
  - *Done when:* running it on reference answers reports near-100% compile rate (sanity check on the harness itself).
- **QC the dataset.** Run every synthesized output through the harness; drop non-compiling rows unless tagged as a deliberate error-to-fix example.
- **Hold out an eval set.** `data/eval.jsonl`, sharing no examples with `data/pairs.jsonl`, weighted toward DeepBook and dependency-pattern tasks. Decontaminate (no eval example appears in training).

**Done when:** `data/eval.jsonl` exists, decontaminated, and the harness runs clean.

---

## 7. Interface contracts (fix before parallel work)

### 7.1 Dataset pair schema (`data/pairs.jsonl`, one JSON object per line)
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

### 7.2 Sharding for Walrus (handoff to training)
- Shard `data/pairs.jsonl` into chunks for Walrus upload (reuse the existing `walrus.rs` path; do not write a second Walrus client). Log each blob id.

### 7.3 Eval report (`eval_report.json`)
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

## 8. Open questions for the human (do not guess)

1. **Dataset public or private?** Affects which teacher tier is acceptable and what can go in `SOURCES.md`. (DeepSeek V4 is MIT, so its output is publishable; confirm intent.)
2. **Teacher access:** DeepSeek V4 via API (Flash tier) or self-hosted? API is simpler and cheaper for synthesis volume.
3. **Final base model:** confirm Gemma 4 E2B, or run a quick bake-off against a small Qwen coder once a first dataset slice exists?
4. **Verify HF repo ids** for Gemma 4 E2B and the DeepSeek V4 teacher tier (record in `training/MODELS.md`).

---

## 9. Start here today

Clone the Sui framework and `deepbookv3`, run the M2 filter (2024 + compile + dedup), and build `scripts/move_eval.py` first. The harness gates everything downstream, so get a baseline number before synthesizing a single pair.

*End of work order. Fix the Section 7 contracts, then work M1 -> M2 -> M5 (harness) before heavy synthesis in M4.*
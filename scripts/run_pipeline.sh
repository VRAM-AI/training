#!/usr/bin/env bash
# Full dataset pipeline: M4 → dep-patterns → merge → M5
# Run via: screen -dmS dataset bash scripts/run_pipeline.sh
# Monitor: screen -r dataset   |   tail -f /tmp/pipeline.log

set -euo pipefail
LOG=/tmp/pipeline.log
exec > >(tee -a "$LOG") 2>&1

REPO=/Users/sidousan/dataset
KEY=$(grep OPENROUTER_API_KEY "$REPO/.env" | cut -d= -f2)
export OPENROUTER_API_KEY="$KEY"

echo "=========================================="
echo "Pipeline start: $(date)"
echo "=========================================="

echo ""
echo "--- Stage M4: synthesis ---"
python3 -u "$REPO/scripts/build_move_dataset.py" --stage m4 --max-per-tag 100
echo "M4 done at $(date). pairs.jsonl lines: $(wc -l < "$REPO/data/pairs.jsonl" 2>/dev/null || echo 0)"

echo ""
echo "--- Stage dep-patterns ---"
python3 -u "$REPO/scripts/gen_dep_patterns.py" --count 40 --out "$REPO/data/dep_patterns.jsonl"
echo "dep-patterns done at $(date). lines: $(wc -l < "$REPO/data/dep_patterns.jsonl" 2>/dev/null || echo 0)"

echo ""
echo "--- Merging dep_patterns into pairs ---"
cat "$REPO/data/dep_patterns.jsonl" >> "$REPO/data/pairs.jsonl"
echo "Merged. Total pairs: $(wc -l < "$REPO/data/pairs.jsonl")"

echo ""
echo "--- Stage M5: QC + eval split ---"
python3 -u "$REPO/scripts/build_move_dataset.py" --stage m5

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "pairs.jsonl : $(wc -l < "$REPO/data/pairs.jsonl") rows"
echo "eval.jsonl  : $(wc -l < "$REPO/data/eval.jsonl" 2>/dev/null || echo 0) rows"
echo "=========================================="

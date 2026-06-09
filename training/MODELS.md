# Model Registry

## Target model (fine-tune + deploy)

| Field | Value |
|---|---|
| Name | Gemma 4 E2B Instruct |
| HF repo (base) | `google/gemma-4-E2B` |
| HF repo (instruct, use this for fine-tune) | `google/gemma-4-E2B-it` |
| License | Apache 2.0 |
| Architecture | Mixture-of-Experts |
| Notes | Browser-deployable via WebGPU/WASM. Training on vram.ai with QLoRA. |

## Teacher model (instruction synthesis)

| Field | Value |
|---|---|
| Name | DeepSeek V4 Flash |
| OpenRouter model ID | `deepseek/deepseek-v4-flash` |
| Fallback (higher quality) | `deepseek/deepseek-v4-pro` |
| License | MIT (outputs redistributable) |
| Access | OpenRouter API (`OPENROUTER_API_KEY` env var) |
| Context window | 1 M tokens |
| Notes | Flash tier preferred for synthesis volume. Switch to Pro only for hard dependency-pattern pairs. |

## Fallback base model

| Field | Value |
|---|---|
| Name | Qwen2.5-Coder-7B-Instruct |
| HF repo | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| License | Apache 2.0 |
| Notes | Use only if Gemma 4 E2B compile-rate < 60% after one training cycle. |

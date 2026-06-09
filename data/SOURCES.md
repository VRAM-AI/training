# Data Sources

All sources are permissive-license. Synthesized outputs use DeepSeek V4 Flash (MIT license).

| Source | URL | License | Contents | Local path |
|---|---|---|---|---|
| Sui monorepo (move-stdlib + sui-framework) | https://github.com/MystenLabs/sui | Apache 2.0 | Move standard library and Sui framework packages | `data/raw/sui/` |
| DeepBook v3 | https://github.com/MystenLabs/deepbookv3 | Apache 2.0 | DeepBook v3 smart contracts, tests, examples | `data/raw/deepbookv3/` |
| Sui examples | https://github.com/MystenLabs/sui/tree/main/examples | Apache 2.0 | Example Move packages | `data/raw/sui/examples/` |
| Sui documentation | https://docs.sui.io | Apache 2.0 | Prose + code snippets (doc-to-code pairs) | `data/raw/sui-docs/` |
| Synthesized pairs | Generated via DeepSeek V4 Flash (OpenRouter) | MIT (model output) | reverse-instruction, error-to-fix, doc-to-code | `data/pairs.jsonl` |

## Notes

- No sources with GPL, AGPL, CC-BY-SA, or unknown licenses are included.
- Every synthesized output is validated by `scripts/move_eval.py` before inclusion.
- Non-compiling synthesized rows are dropped unless tagged `error-to-fix`.

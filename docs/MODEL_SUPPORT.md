# Model support

All real definitions pin a full commit, BF16, deterministic generation,
`local_files_only=True` and `trust_remote_code=False`.

| Key | Model | Revision | Adapter | Current evidence |
| --- | --- | --- | --- | --- |
| `qwen35-0.8b` | Qwen/Qwen3.5-0.8B | `2fc0636…` | Qwen3.5 | Cached on the inspected host; ModelDeck reference supports the runtime shape |
| `qwen35-2b` | Qwen/Qwen3.5-2B | `15852e8…` | Qwen3.5 | Cached; not benchmarked here |
| `qwen35-4b` | Qwen/Qwen3.5-4B | `851bf6e…` | Qwen3.5 | Cached; not benchmarked here |
| `smolvlm2-2.2b` | HuggingFaceTB/SmolVLM2-2.2B-Instruct | `482adb5…` | SmolVLM2 | Cached; processor initialisation verified; adapter unqualified |
| `smolvlm2-500m-video` | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | `7b375e1…` | SmolVLM2 | Cached; processor initialisation verified; adapter unqualified |
| `smolvlm2-256m-video` | HuggingFaceTB/SmolVLM2-256M-Video-Instruct | `067788b…` | SmolVLM2 | Cached; ROCm load and free-text generation verified; structured-output protocol failed |
| `gemma3-4b` | google/gemma-3-4b-it | `093f9f3…` | Gemma 3 | Gated, not cached; adapter unqualified |

“Adapter present” is not a compatibility claim. A model is compatible only after its exact
definition completes the hardware-gated protocol and retained report. An expected class,
missing snapshot, missing dependency, unavailable GPU or generation error becomes a
structured per-model failure and does not prevent later models from being assessed.

Qwen3.5 additionally checks the expected processor and model classes, patch size 16, merge
size 2 and allowlisted visual-token budget. Quantised snapshots are not configured.

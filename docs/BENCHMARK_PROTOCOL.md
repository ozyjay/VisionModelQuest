# Benchmark protocol

Two deterministic contracts are used:

- `scene_json_v1`: strict JSON, at most three objects, one relationship, one uncertainty
  and one safety note; object evidence is at most 15 words and locations use a fixed set;
- `free_text_v1`: no more than three concise plain-text sentences.

Both forbid identity recognition, sensitive-trait inference and invented certainty.

Quick performs one warm-up and two measured requests per selected fixture/question.
Standard performs two warm-ups and ten measured requests per fixture/question. Stability
performs two warm-ups, cycles the same curated tasks for a configurable duration, samples
memory and temperature around every request, then checks unload and process exit.
Comparison selects multiple model keys and executes them sequentially.

Each sample records the environment, exact model identity, adapter, dtype, image dimensions
and bytes, configured visual-token budget and completion limit, load and unload time,
preprocessing, available first-output timing, inference, validation, end-to-end latency,
tokens, throughput, finish reason, contract validity, output hash, host/process/GTT/VRAM
memory fallbacks, peak allocated GPU memory where PyTorch exposes it, temperatures and
failure category. Transformers' non-streaming `generate` cannot expose a reliable
time-to-first-token value, so that field is `null` rather than inferred.

Median and p95 are reported separately; p95 uses nearest rank. Different workloads are not
collapsed into a single score.

## Human review

Create review records with `visionmodelquest review-template`. Score each dimension from
0 (unacceptable) to 4 (excellent/notably reliable): factual correctness, coverage,
hallucination avoidance, counting, spatial accuracy, uncertainty, text reading, concision,
JSON compliance and public safety. Notes explain unusual or non-applicable cases. Human
review is authoritative; automated validation assists only contract and safety checks.


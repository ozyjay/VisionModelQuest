# Benchmark protocol

Two deterministic contracts are used:

- `scene_json_v1`: strict JSON, at most three objects, one relationship, one uncertainty
  and one safety note; object evidence is at most 15 words and locations use a fixed set;
- `free_text_v1`: no more than three concise plain-text sentences.

Both forbid identity recognition, sensitive-trait inference and invented certainty.

The active version-2 workload uses project-created, AI-generated photorealistic PNG fixtures
at 1448 × 1086. Each scene has explicit reference facts, prohibited claims and object labels.
The original 8 × 6 PPM fixtures remain with the version-1 manifest only to reproduce legacy
reports; they are not representative evidence of model vision quality.

Quick performs one warm-up and two measured requests per selected fixture/question.
Standard performs two warm-ups and ten measured requests per fixture/question. Stability
performs two warm-ups, cycles the same curated tasks for a configurable duration, samples
memory and temperature around every request, then checks unload and process exit.
Comparison selects multiple model keys and executes them sequentially.

The parent benchmark process checks available Linux thermal sensors once per second. If
any sensor reaches the default 95 °C safety limit, it immediately kills the isolated model
worker, records a `thermal_limit` failure with the triggering sensor and does not run any
remaining models. Override the limit with `-MaxTemperatureCelsius` when hardware-specific
guidance requires a lower threshold. If the operating system exposes no readable sensors,
this software failsafe cannot operate; firmware thermal protection remains essential.

Each sample records the environment, exact model identity, adapter, dtype, image dimensions
and bytes, configured visual-token budget and completion limit, load and unload time,
preprocessing, available first-output timing, inference, validation, end-to-end latency,
tokens, throughput, finish reason, contract validity, output hash, host/process/GTT/VRAM
memory fallbacks, peak allocated GPU memory where PyTorch exposes it, temperatures and
failure category. Transformers' non-streaming `generate` cannot expose a reliable
time-to-first-token value, so that field is `null` rather than inferred.

Structured-output validation tolerates one optional leading `json` Markdown fence, including
when the model omits only the closing fence. Any trailing content, malformed or incomplete
JSON, extra fields, or schema-bound violation still fails. Failure samples retain generation
measurements and report the JSON location or schema field that caused rejection.

Median and p95 are reported separately; p95 uses nearest rank. Different workloads are not
collapsed into a single score.

## Human review

Create review records with `visionmodelquest review-template`. Score each dimension from
0 (unacceptable) to 4 (excellent/notably reliable): factual correctness, coverage,
hallucination avoidance, counting, spatial accuracy, uncertainty, text reading, concision,
JSON compliance and public safety. Notes explain unusual or non-applicable cases. Human
review is authoritative; automated validation assists only contract and safety checks.

# Results guide

A report is evidence only for its complete environment and model revision. Compare
identical fixture/question selections and visual-token settings.

- Fastest usable: lowest latency among models that pass required contracts and quality.
- Best overall balance: explicit judgement across quality, latency, memory, thermal and
  reliability components; do not invent a hidden composite score.
- Highest quality: highest authoritative human review for the intended workload.
- Lowest memory: lowest comparable peak and retained memory.
- Best structured output: highest strict JSON success rate.
- Unsuitable or incompatible: clear failure category and fingerprint, retained without
  aborting the wider comparison.

Missing sensors, GPU memory counters or time-to-first-output remain `null`. They are not
zero. Thermal readings include their sensor identity. Normal reports cannot be used for
qualitative review because raw model text is intentionally absent; rerun locally with
explicit quality capture.


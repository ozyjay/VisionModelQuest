# VisionModelQuest contributor instructions

VisionModelQuest is a local-first vision-language model evaluation laboratory for the
Framework Desktop. It owns benchmark workloads, direct experimental adapters, measurements,
quality review and compatibility evidence. It does not own model downloads, ModelDeck
lifecycle management, SceneChat camera state or any public UI.

- Use Australian English in prose, comments and user-facing text.
- Benchmark only allowlisted, exact model revisions from the read-only Hugging Face cache.
- Never enable remote code or let a benchmark download model files.
- Keep ordinary tests offline and independent of a GPU, camera, ModelDeck and model weights.
- Treat model output as untrusted. Do not log prompts, images or raw output unless explicit
  local quality-review capture is enabled.
- Run project operations through the PowerShell scripts in `scripts/`.
- Mark physical tests with `hardware`, `rocm`, `large_model` or `long_running`.


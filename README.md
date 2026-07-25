# VisionModelQuest

VisionModelQuest is a local-first command-line laboratory for comparing image-to-text and
vision-language models on the Framework Desktop. It measures compatibility, deterministic
contract following, latency, throughput, memory and thermal observations while keeping
subjective quality under explicit human review.

The benchmark never downloads model weights. Real adapters resolve an allowlisted full
commit from the existing Hugging Face cache with `local_files_only=True` and
`trust_remote_code=False`. Missing, gated or incompatible models produce a structured
failure in the retained report.

## Set up and verify

```powershell
pwsh -NoProfile -File scripts/setup.ps1
pwsh -NoProfile -File scripts/probe_system.ps1
pwsh -NoProfile -File scripts/verify.ps1
pwsh -NoProfile -File scripts/run_benchmark.ps1 -Preset Quick
```

The default setup and Quick run use only the deterministic mock adapter. To create the
separate pinned ROCm environment, use:

```powershell
pwsh -NoProfile -File scripts/setup.ps1 -Rocm
pwsh -NoProfile -File scripts/run_benchmark.ps1 -Preset Quick -Models qwen35-0.8b
```

ROCm setup installs Python packages and AMD's pinned wheels, but no model weights. Prepare
model snapshots separately and explicitly, for example:

```powershell
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
```

Set `VISIONMODELQUEST_HF_CACHE` to the existing Hub cache root when it differs from the
default. Stop an active child model process with:

```powershell
pwsh -NoProfile -File scripts/stop.ps1
```

Reports are timestamped JSON and Markdown files under `reports/`. Normal reports retain
fixture IDs, question IDs, output hashes and measurements, not prompts, images or raw
outputs. `-QualityCapture` is an explicit local-only mode which includes prompt and output
text for review; these reports remain gitignored.

See [architecture](docs/ARCHITECTURE.md), [benchmark protocol](docs/BENCHMARK_PROTOCOL.md),
[model support](docs/MODEL_SUPPORT.md), [results guide](docs/RESULTS_GUIDE.md) and
[ModelDeck integration](docs/MODELDECK_INTEGRATION.md).


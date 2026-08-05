# VisionModelQuest

VisionModelQuest is a local-first vision-language model evaluation laboratory for the
Framework Desktop. It includes two first-class ways to work:

- **Vision Processing Explorer** — a Fedora-native GTK 4 and libadwaita desktop application
  for interactive image, prompt and visual-token experiments.
- **Benchmark suite** — command-line workloads for measuring compatibility, deterministic
  contract following, latency, throughput, memory and thermal observations.

Both keep subjective quality under explicit human review and operate without a browser or
local network service.

The benchmark never downloads model weights. Real adapters resolve an allowlisted full
commit from the existing Hugging Face cache with `local_files_only=True` and
`trust_remote_code=False`. Missing, gated or incompatible models produce a structured
failure in the retained report.

## Launch the desktop application

Set up and open the [Vision Processing Explorer](docs/NATIVE_EXPLORER.md):

```bash
pwsh ./scripts/setup.ps1 -Ui
./scripts/run_ui.sh
```

The application lets you select a local image, load an allowlisted model, edit prompts,
generate output, inspect visual-token regions, manage experiment revisions and review
benchmark reports. For physical inference, also prepare the ROCm environment with
`pwsh ./scripts/setup.ps1 -Rocm`.

## Set up and verify the benchmark suite

```powershell
pwsh -NoProfile -File scripts/setup.ps1
pwsh -NoProfile -File scripts/probe_system.ps1
pwsh -NoProfile -File scripts/verify.ps1
pwsh -NoProfile -File scripts/run_benchmark.ps1 -Preset Quick
```

Benchmarks include an independent thermal failsafe. The model worker is terminated and the
run stops if any available sensor reaches 95 °C. Set a lower hardware-specific limit with
`-MaxTemperatureCelsius`; systems without readable Linux sensors cannot use this failsafe.

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

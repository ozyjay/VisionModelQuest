# Repository inspection and decisions

Inspection was performed against local `main` checkouts matching `origin/main`:

- ModelDeck `1d89298`, including contributor rules, configuration, runtime templates,
  Qwen3.5 and SceneChat workers, cache discovery, hardware probe, benchmark scripts,
  contracts, tests, documentation and recent history;
- SceneChat `b66f562`, including contributor rules, configuration, ModelDeck provider,
  structured schemas and parsing, curated questions, replay assets, hardware acceptance
  tests, documentation and recent history.

## Reusable concepts and boundaries

ModelDeck demonstrates the required trusted-runtime posture: exact cache snapshots,
allowlisted architectures, no remote code, deterministic generation, bounded visual tokens,
isolated inference processes, complete fingerprints, nearest-rank p95 and explicit worker
shutdown. VisionModelQuest reuses these concepts, not its manager, gateway, registry,
downloader, event system or worker ports.

SceneChat supplies the downstream safety boundary: curated questions, strict structured
parsing, generic person language and rejection of identity or sensitive-trait claims.
VisionModelQuest uses a deliberately tighter three-object evaluation contract. It does not
copy SceneChat camera, session, provider, replay or public UI behaviour.

The later integration boundary is ModelDeck's published loopback gateway route. A future
comparison may call that route and measure overhead, but must never call private worker
ports or mutate ModelDeck configuration.

## Dependency risks

- ModelDeck's verified Qwen3.5 environment currently pins ROCm 7.2.1 PyTorch 2.9.1 and
  Transformers 5.13.0. That is evidence for Qwen3.5 on the inspected machine, not proof for
  every candidate.
- SmolVLM2 may require processor or attention behaviour that differs from Qwen3.5.
- Gemma 3 is gated and licence acceptance is required before explicit acquisition.
- Alternative families may conflict with the shared Transformers pin. Their incompatibility
  must be recorded before creating a separate environment.
- `torch.cuda` is the PyTorch ROCm API surface. Its presence alone is not compatibility
  evidence; the probe and a full hardware-gated run are required.

## Phase 1 file decision

Phase 1 contains the package and CLI, allowlisted model configuration, strict contracts,
cache and hardware probes, mock and reviewed Transformers adapters, process-isolated
runner, metrics/reporting/review workflow, versioned fixtures, PowerShell operations,
documentation and offline tests. Physical evidence remains a separate opt-in activity.


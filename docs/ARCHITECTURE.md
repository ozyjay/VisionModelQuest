# Architecture

The CLI validates only version-controlled model, fixture and question definitions. It runs
selected models sequentially. Each model receives one isolated child process, loads once,
executes its workload, unloads and exits before the next model begins.

```text
PowerShell script
  -> control CLI and environment fingerprint
    -> allowlisted worker subprocess
      -> read-only pinned Hugging Face snapshot
      -> curated local fixtures and questions
      -> deterministic adapter generation
    <- structured measurements and hashes
  -> timestamped JSON and Markdown reports
```

The child command is constructed internally from validated model keys and fixed CLI
arguments. Configuration cannot provide Python classes, commands, arbitrary paths or
environment variables. The adapter registry is a Python allowlist.

The model process is the memory-recovery boundary. Timeout or interruption terminates its
process group, and `var/active-worker.pid` lets the stop script target only the validated
active worker. The control environment contains tests and reporting dependencies; the
ROCm environment contains the direct experimental model stack. Neither replaces Fedora
ROCm packages or ModelDeck's environments.

The normal privacy boundary is the sample serialiser. It keeps image dimensions and size,
fixture/question IDs, output hashes, token counts and hardware measurements. Explicit
quality capture adds prompts and raw outputs to local gitignored reports. Image bytes are
never written to reports.


# Native Vision Processing Explorer

VisionModelQuest includes a Fedora-native GTK 4 and libadwaita application for local,
interactive image-to-text experiments. It does not run a web server, open a network port or
download model files.

## Set up and launch

The interface uses Fedora's system GTK bindings through a dedicated virtual environment.
The project's pure-Python runtime dependencies are reused from the control environment;
Torch and Transformers remain confined to the ROCm worker environment.

```bash
pwsh ./scripts/setup.ps1 -Ui
./scripts/run_ui.sh
```

For physical inference, prepare the ROCm environment separately:

```bash
pwsh ./scripts/setup.ps1 -Rocm
```

The interface starts the selected processor without loading weights. Select **Load Model**
before generation. **Cancel** terminates the worker process group and creates a clean,
processor-only replacement.

Run the display-free suite through the standard verification command:

```bash
pwsh ./scripts/verify.ps1
```

Run GTK checks from an active Wayland or X11 session:

```bash
./scripts/verify_ui.sh
```

Stop both the application and worker using validated PID records:

```bash
pwsh ./scripts/stop.ps1
```

Optional user-local desktop integration is available through:

```bash
./scripts/install_desktop_entry.sh
```

This writes only beneath the current user's XDG data directory.

## Isolation and storage

The GTK process communicates with one inference worker using versioned JSON-lines over
standard input and output. Requests accept allowlisted model keys and bounded values only.
Images are decoded and validated, then copied into an application-owned session directory
before the worker sees an opaque image ID.

Persistent state is stored beneath:

```text
$XDG_STATE_HOME/visionmodelquest/
├── assets/sha256/
├── experiments/
└── logs/
```

Temporary sessions and PID records are stored beneath:

```text
$XDG_RUNTIME_DIR/visionmodelquest/
├── application.pid
├── worker.pid
└── sessions/
```

Unsaved session images and raw generation output are removed when the application closes.
Ordinary experiment revisions retain hashes, token counts, timings, validation state and
preprocessing metadata, not raw output.

## Visual-token interpretation

The Qwen3.5 inspector exposes preprocessing geometry rather than a claim about internal
meaning. For the checked-in 8 × 6 fixtures and the allowlisted budget of 140, preprocessing
produces a 224 × 320 image, a 14 × 20 raw patch grid and a 7 × 10 merged grid containing 70
visual tokens.

Visual-token regions are spatial inputs to the model. They are not semantic miniature
images and do not show what the model has understood.

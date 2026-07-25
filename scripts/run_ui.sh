#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -x .venv-ui/bin/python ]]; then
    printf '%s\n' 'Missing .venv-ui. Run: pwsh ./scripts/setup.ps1 -Ui'
    exit 2
fi
if [[ ! -x .venv-rocm72/bin/python ]]; then
    printf '%s\n' 'Warning: .venv-rocm72 is missing; only the mock worker can run.' >&2
fi
if ! command -v glib-compile-schemas >/dev/null 2>&1; then
    printf '%s\n' 'glib-compile-schemas is required to launch the native interface.' >&2
    exit 2
fi

schema_dir="$project_root/build/schemas"
mkdir -p "$schema_dir"
cp "$project_root/data/io.github.ozyjay.VisionModelQuest.gschema.xml" "$schema_dir/"
glib-compile-schemas --strict "$schema_dir"

export GSETTINGS_SCHEMA_DIR="$schema_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$project_root/.venv-ui/bin/python" -m visionmodelquest.ui.application "$@"

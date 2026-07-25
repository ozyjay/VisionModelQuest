#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -x .venv-ui/bin/python ]]; then
    printf '%s\n' 'Missing .venv-ui. Run: pwsh ./scripts/setup.ps1 -Ui'
    exit 2
fi
if [[ -z "${WAYLAND_DISPLAY:-}" && -z "${DISPLAY:-}" ]]; then
    printf '%s\n' 'UI verification requires an active Wayland or X11 session.'
    exit 2
fi

.venv-ui/bin/python -m pytest -m gtk

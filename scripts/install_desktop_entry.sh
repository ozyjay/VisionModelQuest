#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
applications_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icons_dir="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
mkdir -p "$applications_dir" "$icons_dir"
sed "s|@PROJECT_ROOT@|$project_root|g" \
    "$project_root/data/io.github.ozyjay.VisionModelQuest.desktop.in" \
    > "$applications_dir/io.github.ozyjay.VisionModelQuest.desktop"
cp "$project_root/data/io.github.ozyjay.VisionModelQuest.svg" \
    "$icons_dir/io.github.ozyjay.VisionModelQuest.svg"
printf '%s\n' 'Installed user-local Vision Processing Explorer desktop integration.'

#!/usr/bin/env bash
# Preflight checks before pushing this project to GitHub.
# - Finds large files that may exceed GitHub limits
# - Highlights generated folders that should stay out of git

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== GitHub Preflight ==="
echo "Repo path: $SCRIPT_DIR"
echo

echo "[1/2] Large files (>50 MB):"
LARGE_FILES=$(find . -type f -size +50M -not -path './.git/*' -printf '%s %p\n' | sort -nr || true)
if [[ -z "$LARGE_FILES" ]]; then
    echo "  None found."
else
    echo "$LARGE_FILES" | awk '{printf "  - %.1f MB %s\n", $1/1024/1024, $2}'
fi

echo
echo "[2/2] Generated directories present:"
for dir in output training_output test_training wheels venv .venv; do
    if [[ -d "$dir" ]]; then
        SIZE=$(du -sh "$dir" | cut -f1)
        echo "  - $dir ($SIZE)"
    fi
done

echo
echo "If these are local-only artifacts, keep them untracked (already covered by .gitignore)."

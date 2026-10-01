#!/bin/sh
# Enables Baton's versioned git hooks for this clone (Linux / macOS).
# Run once after cloning:   sh scripts/install-hooks.sh
set -eu

cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
echo "Git hooks enabled (core.hooksPath = .githooks)."
echo "Every commit is now scanned for secrets by scripts/secret_scan.py."

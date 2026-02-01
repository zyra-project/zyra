#!/usr/bin/env bash
set -euo pipefail

# Identify yourself
git config --global user.name "Eric Hackathorn"
git config --global user.email "erichackathorn@gmail.com"

# Git LFS (skip smudge to avoid 404s on checkout)
git lfs install --skip-smudge
git config filter.lfs.smudge "git-lfs smudge --skip -- %f"
git config filter.lfs.required true

# Ensure remotes (reset origin if already exists)
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/zyra-project/zyra.git

git remote remove upstream 2>/dev/null || true
git remote add upstream https://github.com/NOAA-GSL/zyra.git

git fetch origin --prune || true
git fetch upstream --prune || true

# Switch to mirror/staging if available
if git show-ref --quiet refs/remotes/origin/mirror/staging; then
  git checkout mirror/staging
fi

# Install Poetry + dependencies if pyproject.toml exists
if ! command -v poetry >/dev/null 2>&1; then
  pip install poetry
fi
if [ -f pyproject.toml ]; then
  poetry install --with dev --all-extras
fi

echo "✅ setup.sh complete: Git, LFS, remotes, and dependencies installed"

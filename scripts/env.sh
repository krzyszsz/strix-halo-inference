#!/usr/bin/env bash
# Shared environment defaults for all local scripts.

_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ENV_REPO_ROOT="$(cd "$_ENV_SCRIPT_DIR/.." && pwd)"

REPO_ROOT="${REPO_ROOT:-$_ENV_REPO_ROOT}"
HF_ROOT="${HF_ROOT:-/mnt/hf}"
MODEL_ROOT="${MODEL_ROOT:-$HF_ROOT/models}"
# Keep HF tokens outside the repo for public publishing. Default to "$HOME/hf.key".
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/hf.key}"

export REPO_ROOT
export HF_ROOT
export MODEL_ROOT
export HF_TOKEN_FILE

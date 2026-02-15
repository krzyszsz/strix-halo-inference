#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

TARGET_DIR="${TARGET_DIR:-$MODEL_ROOT/vision-face}"

ULTRAFACE_REPO="${ULTRAFACE_REPO:-onnxmodelzoo/version-RFB-320}"
ULTRAFACE_FILE="${ULTRAFACE_FILE:-version-RFB-320.onnx}"

ARCFACE_REPO="${ARCFACE_REPO:-onnxmodelzoo/arcfaceresnet100-8}"
ARCFACE_FILE="${ARCFACE_FILE:-arcfaceresnet100-8.onnx}"

mkdir -p "$TARGET_DIR/ultraface" "$TARGET_DIR/arcface"

echo "Downloading UltraFace detector..."
hf download "$ULTRAFACE_REPO" --repo-type model --local-dir "$TARGET_DIR/ultraface" --include "$ULTRAFACE_FILE"

echo "Downloading ArcFace recognizer..."
hf download "$ARCFACE_REPO" --repo-type model --local-dir "$TARGET_DIR/arcface" --include "$ARCFACE_FILE"

echo "Done: $TARGET_DIR"


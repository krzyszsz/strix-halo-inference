#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

MODEL_REPO="${MODEL_REPO:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
TARGET_DIR="${TARGET_DIR:-$MODEL_ROOT/voxtral-mini-4b-realtime-2602-hf}"

if ! mkdir -p "$TARGET_DIR" 2>/dev/null; then
  sudo mkdir -p "$TARGET_DIR"
fi
if [ ! -w "$TARGET_DIR" ]; then
  sudo chown -R "$(id -u):$(id -g)" "$TARGET_DIR"
fi
if [ ! -w "$(dirname "$TARGET_DIR")" ]; then
  sudo chown "$(id -u):$(id -g)" "$(dirname "$TARGET_DIR")"
fi

if command -v hf >/dev/null 2>&1; then
  # Public repo; no token required. Do not pass tokens on the command line.
  hf download "$MODEL_REPO" --repo-type model --local-dir "$TARGET_DIR"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_REPO" \
    --local-dir "$TARGET_DIR" \
    --local-dir-use-symlinks False
else
  echo "ERROR: missing Hugging Face CLI ('hf' or 'huggingface-cli')." >&2
  exit 2
fi

TARGET_DIR="$TARGET_DIR" python - <<'PY'
import json
import os
from pathlib import Path

target = Path(os.environ["TARGET_DIR"])
params_path = target / "params.json"

params = json.loads(params_path.read_text(encoding="utf-8"))
mm = params.get("multimodal", {})
whisper_args = (mm.get("whisper_model_args") or {}).get("encoder_args") or {}
audio_enc = whisper_args.get("encoder_args") or whisper_args
audio_encoding = (audio_enc.get("audio_encoding_args") or {})

def pick(name: str, fallback):
    value = params.get(name, fallback)
    return fallback if value is None else value

def pick_audio(name: str, fallback):
    value = audio_enc.get(name, fallback)
    return fallback if value is None else value

hidden_size = int(pick("dim", 3072))
vocab_size = int(pick("vocab_size", 131072))
num_layers = int(pick("n_layers", 26))
num_heads = int(pick("n_heads", 32))
num_kv_heads = int(pick("n_kv_heads", 8))
head_dim = int(pick("head_dim", 128))
intermediate_size = int(pick("hidden_dim", 9216))
rope_theta = float(pick("rope_theta", 1_000_000.0))
sliding_window = pick("sliding_window", 8192)
model_max_length = int(pick("model_max_length", 131072))

audio_hidden_size = int(pick_audio("dim", 1280))
audio_layers = int(pick_audio("n_layers", 32))
audio_heads = int(pick_audio("n_heads", 32))
audio_kv_heads = int(pick_audio("n_kv_heads", audio_heads))
audio_head_dim = int(pick_audio("head_dim", 64))
audio_intermediate = int(pick_audio("hidden_dim", 5120))

num_mel_bins = int(audio_encoding.get("num_mel_bins", 128))
sampling_rate = int(audio_encoding.get("sampling_rate", 16000))
hop_length = int(audio_encoding.get("hop_length", 160))
window_size = int(audio_encoding.get("window_size", 400))

# HF-style config based on Voxtral-Mini-3B-2507 schema, adjusted from params.json.
config = {
    "architectures": ["VoxtralForConditionalGeneration"],
    "audio_config": {
        "activation_dropout": 0.0,
        "activation_function": "gelu",
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "head_dim": audio_head_dim,
        "hidden_size": audio_hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": audio_intermediate,
        "layerdrop": 0.0,
        "max_source_positions": 1500,
        "model_type": "voxtral_encoder",
        "num_attention_heads": audio_heads,
        "num_hidden_layers": audio_layers,
        "num_key_value_heads": audio_kv_heads,
        "num_mel_bins": num_mel_bins,
        "scale_embedding": False,
        "vocab_size": vocab_size,
    },
    "audio_token_id": 24,
    "hidden_size": hidden_size,
    "model_type": "voxtral",
    "projector_hidden_act": "gelu",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": model_max_length,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": rope_theta,
        "sliding_window": sliding_window,
        "use_cache": True,
        "vocab_size": vocab_size,
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "unknown",
    "vocab_size": vocab_size,
}
(target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

preproc = {
    "chunk_length": 30,
    "dither": 0.0,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": num_mel_bins,
    "hop_length": hop_length,
    "n_fft": window_size,
    "n_samples": sampling_rate * 30,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "VoxtralProcessor",
    "return_attention_mask": False,
    "sampling_rate": sampling_rate,
}
(target / "preprocessor_config.json").write_text(json.dumps(preproc, indent=2), encoding="utf-8")

gen_cfg = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 11,
    "transformers_version": "unknown",
}
(target / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2), encoding="utf-8")

src = target / "consolidated.safetensors"
dst = target / "model.safetensors"
if src.exists() and not dst.exists():
    # Relative symlink keeps the folder portable if moved as a unit.
    dst.symlink_to(src.name)
PY

echo "$TARGET_DIR"

#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from transformers.processing_utils import ProcessorMixin


def main() -> int:
    model_id = os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-3B-2507")
    input_audio = Path(os.environ["INPUT_AUDIO"])
    clip_seconds = float(os.environ.get("CLIP_SECONDS", "30"))
    out_text = Path(os.environ["OUT_TEXT"])
    out_json = Path(os.environ["OUT_JSON"])

    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def rel_for_report(path: Path) -> str:
        repo_root = os.environ.get("REPO_ROOT")
        candidates = []
        if repo_root:
            candidates.append(Path(repo_root).resolve())
        candidates.append(Path.cwd().resolve())
        rp = path.resolve()
        for base in candidates:
            try:
                return str(rp.relative_to(base))
            except Exception:
                continue
        return str(path)

    wave, sr = sf.read(input_audio)
    max_samples = int(sr * clip_seconds)
    if len(wave) > max_samples:
        wave = wave[:max_samples]
    clip_path = out_text.with_suffix(".clip.wav")
    sf.write(clip_path, wave, sr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.time()
    # Work around a transformers repr bug triggered during processor loading for Voxtral configs.
    ProcessorMixin.__repr__ = lambda self: f"{self.__class__.__name__}()"
    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=("cuda" if device == "cuda" else "cpu"),
    )

    inputs = processor.apply_transcription_request(
        language="en",
        audio=str(clip_path),
        model_id=model_id,
    )
    if device == "cuda":
        inputs = inputs.to(device, dtype=dtype)
    else:
        inputs = inputs.to(device)

    outputs = model.generate(**inputs, max_new_tokens=512)
    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    transcript = decoded[0] if decoded else ""

    elapsed = time.time() - t0
    out_text.write_text(transcript + "\n", encoding="utf-8")

    summary = {
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        "input_audio": rel_for_report(input_audio),
        "clip_audio": rel_for_report(clip_path),
        "clip_seconds": clip_seconds,
        "elapsed_seconds": round(elapsed, 2),
        "transcript_chars": len(transcript),
        "out_text": rel_for_report(out_text),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_json)
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

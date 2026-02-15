#!/usr/bin/env python3
import json
import os
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def main() -> int:
    model_id = os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
    input_audio = Path(os.environ["INPUT_AUDIO"])
    out_txt = Path(os.environ["OUT_TXT"])
    out_json = Path(os.environ["OUT_JSON"])
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))
    language = os.environ.get("LANGUAGE", "").strip() or None
    local_files_only = os.environ.get("LOCAL_FILES_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    force_cpu = os.environ.get("FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model_id": model_id,
        "input_audio": str(input_audio),
        "status": "unknown",
        "device_available": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "hip_version": getattr(torch.version, "hip", None),
        "force_cpu": force_cpu,
        "local_files_only": local_files_only,
        "torch_version": torch.__version__,
        "max_new_tokens": max_new_tokens,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    try:
        # Workaround for a transformers bug: some tokenizers (including Mistral/Tekken) do not
        # implement `added_tokens_decoder`, but processor initialization may call `__repr__`,
        # which assumes it exists.
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        PreTrainedTokenizerBase.added_tokens_decoder = property(lambda self: {})  # type: ignore[assignment]

        t0 = time.time()
        _log(f"loading processor from {model_id} (local_files_only={local_files_only}) ...")
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        use_cuda = torch.cuda.is_available() and not force_cpu
        dtype = torch.bfloat16 if use_cuda else torch.float32

        # Keep this explicit to match the other probes in this repo.
        device_map = "cuda" if use_cuda else "cpu"
        _log(f"loading model (dtype={dtype}, device_map={device_map}) ...")
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=local_files_only,
        )
        model.eval()

        _log(f"building transcription request (language={language or 'auto'}) ...")
        inputs = processor.apply_transcription_request(
            audio=str(input_audio),
            model_id=model_id,
            language=language,
        )
        if use_cuda:
            inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}

        _log(f"running generate(max_new_tokens={max_new_tokens}) ...")
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        _log("decoding output ...")
        # VoxtralProcessor inherits ProcessingMixin, but decoding is tokenizer-driven.
        text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        out_txt.write_text(text + "\n", encoding="utf-8")
        _log("done")

        result["status"] = "ok"
        result["elapsed_seconds"] = round(time.time() - t0, 2)
        result["transcript_preview"] = text[:200]
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_json)
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import json
import os
import time
import traceback
from pathlib import Path

import torch


def main() -> int:
    model_id = os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
    local_files_only = os.environ.get("LOCAL_FILES_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    if not local_files_only:
        try:
            local_files_only = Path(model_id).exists()
        except Exception:
            local_files_only = False

    force_cpu = os.environ.get("FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
    out_json = Path(os.environ["OUT_JSON"])
    out_json.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model_id": model_id,
        "status": "unknown",
        "device_available": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "hip_version": getattr(torch.version, "hip", None),
        "force_cpu": force_cpu,
        "local_files_only": local_files_only,
        "torch_version": torch.__version__,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    try:
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        t0 = time.time()
        try:
            _processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        except NotImplementedError:
            # Workaround for a transformers bug: some tokenizers (including Mistral/Tekken) do not
            # implement `added_tokens_decoder`, but the processor loader may call `__repr__`
            # which assumes it exists. Returning an empty mapping is sufficient for this probe.
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase

            PreTrainedTokenizerBase.added_tokens_decoder = property(lambda self: {})  # type: ignore[assignment]
            _processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        result["processor_loaded"] = True
        use_cuda = torch.cuda.is_available() and not force_cpu
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "cuda" if use_cuda else "cpu"
        _model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=local_files_only,
        )
        result["model_loaded"] = True
        result["status"] = "loaded"
        result["elapsed_seconds"] = round(time.time() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_json)

    return 0 if result["status"] == "loaded" else 2


if __name__ == "__main__":
    raise SystemExit(main())

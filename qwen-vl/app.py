import base64
import io
import os
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:  # fallback for older transformers
    from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
from qwen_vl_utils import process_vision_info


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
DTYPE = os.getenv("DTYPE", "float16")
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
ATTN_IMPL = os.getenv("ATTN_IMPL", "sdpa")
USE_FAST_PROCESSOR = os.getenv("USE_FAST_PROCESSOR", "true").lower() in {"1", "true", "yes"}
MIN_PIXELS = int(os.getenv("MIN_PIXELS", "3136"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "12582912"))

app = FastAPI()


class VisionParameters(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20


class VisionRequest(BaseModel):
    prompt: str = Field(..., description="Text instruction for the vision model")
    image_b64: str = Field(..., description="Base64-encoded image")
    parameters: VisionParameters = VisionParameters()


_model = None
_processor = None


def _torch_dtype(dtype: str) -> torch.dtype:
    dt = dtype.lower()
    if dt in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dt in {"fp32", "float32", "f32"}:
        return torch.float32
    return torch.float16


def _load_model() -> None:
    global _model, _processor
    if _model is not None:
        return

    torch_dtype = _torch_dtype(DTYPE)
    try:
        _model = QwenVLModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=DEVICE_MAP,
            attn_implementation=ATTN_IMPL,
        )
    except Exception:
        _model = QwenVLModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map=DEVICE_MAP,
            attn_implementation=ATTN_IMPL,
        )
    _processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        use_fast=USE_FAST_PROCESSOR,
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_id": MODEL_ID}


@app.post("/v1/vision/describe")
def describe(req: VisionRequest) -> Dict[str, Any]:
    _load_model()

    try:
        image_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": req.prompt},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": req.parameters.max_new_tokens,
        "temperature": req.parameters.temperature,
        "top_p": req.parameters.top_p,
        "top_k": req.parameters.top_k,
    }

    output_ids = _model.generate(**inputs, **gen_kwargs)
    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = _processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {
        "model": MODEL_ID,
        "text": output_text,
    }

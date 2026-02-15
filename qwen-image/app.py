import base64
import io
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import login


app = FastAPI(title="Qwen-Image API", version="1.0")

HF_ROOT = os.environ.get("HF_ROOT", "/mnt/hf")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HF_ROOT, "models"))
DEFAULT_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen-Image")
DTYPE = os.environ.get("DTYPE", "bfloat16")
DEVICE = os.environ.get("DEVICE", "cuda")

_PIPELINES: Dict[Tuple[str, str], DiffusionPipeline] = {}


class GenerationRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    model: str
    created: int
    data: list


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype.lower() in {"fp32", "float32", "f32"}:
        return torch.float32
    return torch.float16


def _resolve_model_id(model_id: Optional[str]) -> str:
    model_id = (model_id or DEFAULT_MODEL).strip()
    if os.path.exists(model_id):
        return model_id
    base = model_id.split("/")[-1]
    candidates = [
        base,
        base.lower(),
        base.replace("_", "-").lower(),
        base.replace(".", "-").lower(),
    ]
    for name in candidates:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            return path
    return model_id


def _pipeline_key(model_id: str, dtype: str) -> Tuple[str, str]:
    return (model_id, dtype.lower())


def _get_pipeline(model_id: str, dtype: str) -> DiffusionPipeline:
    key = _pipeline_key(model_id, dtype)
    if key in _PIPELINES:
        return _PIPELINES[key]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype(dtype),
    )

    if os.environ.get("VAE_SLICING", "1") == "1" and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if os.environ.get("VAE_TILING", "0") == "1" and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    pipe.to(DEVICE)
    _PIPELINES[key] = pipe
    return pipe


def _should_fallback_fp32(exc: Exception) -> bool:
    msg = str(exc)
    return "expected scalar type Float but found Half" in msg


def _parse_size(size: Optional[str]) -> Dict[str, int]:
    if not size:
        return {}
    if "x" not in size:
        return {}
    w, h = size.lower().split("x", 1)
    try:
        return {"width": int(w), "height": int(h)}
    except ValueError:
        return {}


def _extract_kwargs(parameters: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if not parameters:
        return kwargs

    if "num_inference_steps" in parameters:
        kwargs["num_inference_steps"] = int(parameters["num_inference_steps"])
    if "guidance_scale" in parameters:
        kwargs["guidance_scale"] = float(parameters["guidance_scale"])
    if "true_cfg_scale" in parameters:
        kwargs["true_cfg_scale"] = float(parameters["true_cfg_scale"])
    if "height" in parameters:
        kwargs["height"] = int(parameters["height"])
    if "width" in parameters:
        kwargs["width"] = int(parameters["width"])
    if "negative_prompt" in parameters:
        kwargs["negative_prompt"] = parameters["negative_prompt"]

    n_images = parameters.get("n") or parameters.get("num_images")
    if n_images:
        kwargs["num_images_per_prompt"] = int(n_images)

    size = parameters.get("size")
    if size and ("height" not in kwargs or "width" not in kwargs):
        kwargs.update(_parse_size(str(size)))

    seed = parameters.get("seed")
    if seed is not None:
        kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(int(seed))

    return kwargs


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/images/generations", response_model=GenerationResponse)
def generate(request: GenerationRequest) -> GenerationResponse:
    model_id = _resolve_model_id(request.model)
    dtype = DTYPE
    try:
        pipe = _get_pipeline(model_id, dtype)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    kwargs = _extract_kwargs(request.parameters)

    try:
        with torch.inference_mode():
            result = pipe(request.prompt, **kwargs)
    except Exception as exc:
        if dtype.lower() in {"float16", "fp16", "f16", "half"} and _should_fallback_fp32(exc):
            fallback_dtype = "float32"
            try:
                pipe = _get_pipeline(model_id, fallback_dtype)
                with torch.inference_mode():
                    result = pipe(request.prompt, **kwargs)
            except Exception as exc2:
                raise HTTPException(status_code=500, detail=f"Inference failed (fp32 fallback): {exc2}") from exc2
        else:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    images = result.images or []
    data = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})

    return GenerationResponse(model=model_id, created=int(time.time()), data=data)

import base64
import io
import inspect
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import AutoPipelineForText2Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import login


app = FastAPI(title="Stable Diffusion API", version="1.0")

HF_ROOT = os.environ.get("HF_ROOT", "/mnt/hf")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HF_ROOT, "models"))
DEFAULT_MODEL = os.environ.get("MODEL_ID", os.path.join(MODEL_DIR, "sd35-medium"))
DTYPE = os.environ.get("DTYPE", "float16")
DEVICE = os.environ.get("DEVICE", "cuda")
ENABLE_MODEL_CPU_OFFLOAD = os.environ.get("MODEL_CPU_OFFLOAD", "0") == "1"
ENABLE_SEQUENTIAL_CPU_OFFLOAD = os.environ.get("SEQUENTIAL_CPU_OFFLOAD", "0") == "1"
ENABLE_ATTN_SLICING = os.environ.get("ATTN_SLICING", "0") == "1"
DEVICE_MAP = os.environ.get("DEVICE_MAP", "").strip()
MAX_GPU_MEMORY = os.environ.get("MAX_GPU_MEMORY", "").strip()
MAX_CPU_MEMORY = os.environ.get("MAX_CPU_MEMORY", "").strip()

_PIPELINES: Dict[Tuple[str, str], AutoPipelineForText2Image] = {}


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


def _from_pretrained_kwargs(dtype_value: torch.dtype) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"torch_dtype": dtype_value}
    if DEVICE_MAP:
        kwargs["device_map"] = DEVICE_MAP
        max_memory: Dict[Any, str] = {}
        if MAX_GPU_MEMORY:
            max_memory[0] = MAX_GPU_MEMORY
        if MAX_CPU_MEMORY:
            max_memory["cpu"] = MAX_CPU_MEMORY
        if max_memory:
            kwargs["max_memory"] = max_memory
    return kwargs


def _get_pipeline(model_id: str, dtype: str) -> AutoPipelineForText2Image:
    key = _pipeline_key(model_id, dtype)
    if key in _PIPELINES:
        return _PIPELINES[key]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    dtype_value = _torch_dtype(dtype)
    try:
        load_kwargs = _from_pretrained_kwargs(dtype_value)
        uses_device_map = "device_map" in load_kwargs
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if os.environ.get("VAE_SLICING", "1") == "1" and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if os.environ.get("VAE_TILING", "0") == "1" and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if ENABLE_ATTN_SLICING and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        if ENABLE_SEQUENTIAL_CPU_OFFLOAD and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif ENABLE_MODEL_CPU_OFFLOAD and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif not uses_device_map:
            pipe.to(DEVICE)
    except Exception as exc:
        # Work around a diffusers/torch meta-tensor load path observed for SD3.5-medium on ROCm.
        if "Cannot copy out of meta tensor" not in str(exc):
            raise
        load_kwargs = _from_pretrained_kwargs(dtype_value)
        uses_device_map = "device_map" in load_kwargs
        load_kwargs["low_cpu_mem_usage"] = False
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if os.environ.get("VAE_SLICING", "1") == "1" and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if os.environ.get("VAE_TILING", "0") == "1" and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if ENABLE_ATTN_SLICING and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        if ENABLE_SEQUENTIAL_CPU_OFFLOAD and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif ENABLE_MODEL_CPU_OFFLOAD and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif not uses_device_map:
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


def _filter_supported_kwargs(pipe: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


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
    kwargs = _filter_supported_kwargs(pipe, kwargs)

    try:
        with torch.inference_mode():
            result = pipe(prompt=request.prompt, **kwargs)
    except Exception as exc:
        if dtype.lower() in {"float16", "fp16", "f16", "half"} and _should_fallback_fp32(exc):
            fallback_dtype = "float32"
            try:
                pipe = _get_pipeline(model_id, fallback_dtype)
                with torch.inference_mode():
                    result = pipe(prompt=request.prompt, **kwargs)
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

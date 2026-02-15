import base64
import inspect
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import login
from PIL import Image

try:
    from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
except Exception:  # pragma: no cover
    QwenImageEditPipeline = None
    QwenImageEditPlusPipeline = None


app = FastAPI(title="Qwen-Image-Edit API", version="1.0")

HF_ROOT = os.environ.get("HF_ROOT", "/mnt/hf")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HF_ROOT, "models"))
DEFAULT_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit")
DTYPE = os.environ.get("DTYPE", "bfloat16")
DEVICE = os.environ.get("DEVICE", "cuda")

_PIPELINES: Dict[Tuple[str, str], Any] = {}


class EditRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    image: Optional[str] = None
    image_b64: Optional[str] = None
    images: Optional[List[str]] = None
    images_b64: Optional[List[str]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EditResponse(BaseModel):
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


def _pipeline_class(model_id: str):
    model_lower = model_id.lower()
    if "2511" in model_lower or "2509" in model_lower:
        if QwenImageEditPlusPipeline is None:
            raise RuntimeError("QwenImageEditPlusPipeline not available; update diffusers")
        return QwenImageEditPlusPipeline
    if QwenImageEditPipeline is None:
        raise RuntimeError("QwenImageEditPipeline not available; update diffusers")
    return QwenImageEditPipeline


def _pipeline_key(model_id: str, dtype: str) -> Tuple[str, str]:
    return (model_id, dtype.lower())


def _is_plus_model(model_id: str) -> bool:
    model_lower = model_id.lower()
    return "2511" in model_lower or "2509" in model_lower


def _get_pipeline(model_id: str, dtype: str):
    key = _pipeline_key(model_id, dtype)
    if key in _PIPELINES:
        return _PIPELINES[key]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    pipeline_cls = _pipeline_class(model_id)
    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype(dtype),
    )

    if os.environ.get("VAE_SLICING", "1") == "1" and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if os.environ.get("VAE_TILING", "0") == "1" and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if os.environ.get("ATTENTION_SLICING", "0") == "1" and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")

    if os.environ.get("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "0") == "1" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload(device=DEVICE)
    elif os.environ.get("ENABLE_MODEL_CPU_OFFLOAD", "0") == "1" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(device=DEVICE)
    else:
        pipe.to(DEVICE)
    _PIPELINES[key] = pipe
    return pipe


def _should_fallback_fp32(exc: Exception) -> bool:
    msg = str(exc)
    return "expected scalar type Float but found Half" in msg


def _decode_image(image_b64: str) -> Image.Image:
    if image_b64.startswith("data:"):
        _, image_b64 = image_b64.split(",", 1)
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _collect_image_inputs(request: EditRequest) -> List[str]:
    inputs: List[str] = []
    if request.image:
        inputs.append(request.image)
    if request.image_b64:
        inputs.append(request.image_b64)
    if request.images:
        inputs.extend([x for x in request.images if x])
    if request.images_b64:
        inputs.extend([x for x in request.images_b64 if x])
    return [x for x in inputs if x]


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
    if "strength" in parameters:
        kwargs["strength"] = float(parameters["strength"])
    if "height" in parameters:
        kwargs["height"] = int(parameters["height"])
    if "width" in parameters:
        kwargs["width"] = int(parameters["width"])
    if "negative_prompt" in parameters:
        kwargs["negative_prompt"] = parameters["negative_prompt"]
    if "max_sequence_length" in parameters:
        kwargs["max_sequence_length"] = int(parameters["max_sequence_length"])
    if "num_images_per_prompt" in parameters:
        kwargs["num_images_per_prompt"] = int(parameters["num_images_per_prompt"])

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


@app.post("/v1/images/edits", response_model=EditResponse)
def edit(request: EditRequest) -> EditResponse:
    image_inputs = _collect_image_inputs(request)
    if not image_inputs:
        raise HTTPException(status_code=400, detail="Missing image input. Provide image/image_b64 or images/images_b64.")
    if len(image_inputs) > 3:
        raise HTTPException(status_code=400, detail="At most 3 input images are supported.")

    model_id = _resolve_model_id(request.model)
    if len(image_inputs) > 1 and not _is_plus_model(model_id):
        raise HTTPException(
            status_code=400,
            detail="Multi-image input requires Qwen-Image-Edit-2509/2511 (Plus pipeline).",
        )

    decoded_images: List[Image.Image] = []
    for idx, image_b64 in enumerate(image_inputs):
        try:
            decoded_images.append(_decode_image(image_b64))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image data at index {idx}: {exc}") from exc

    image_arg: Any = decoded_images[0] if len(decoded_images) == 1 else decoded_images

    dtype = DTYPE
    try:
        pipe = _get_pipeline(model_id, dtype)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    kwargs = _extract_kwargs(request.parameters)
    kwargs = _filter_supported_kwargs(pipe, kwargs)

    try:
        with torch.inference_mode():
            result = pipe(prompt=request.prompt, image=image_arg, **kwargs)
    except Exception as exc:
        if dtype.lower() in {"float16", "fp16", "f16", "half"} and _should_fallback_fp32(exc):
            fallback_dtype = "float32"
            try:
                pipe = _get_pipeline(model_id, fallback_dtype)
                with torch.inference_mode():
                    result = pipe(prompt=request.prompt, image=image_arg, **kwargs)
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

    return EditResponse(model=model_id, created=int(time.time()), data=data)

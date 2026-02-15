#!/usr/bin/env python3
import io
import inspect
import os
import time
from pathlib import Path
from typing import List

import requests
import torch
from diffusers import Flux2Pipeline
from huggingface_hub import get_token
from PIL import Image


def log(message: str) -> None:
    print(message, flush=True)


def getenv_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def getenv_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def getenv_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def torch_dtype_from_env(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key in {"fp16", "float16", "f16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "f32"}:
        return torch.float32
    return torch.bfloat16


def filter_supported_kwargs(pipe: Flux2Pipeline, kwargs: dict) -> dict:
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def parse_image_refs(raw: str) -> List[Image.Image]:
    refs: List[Image.Image] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        refs.append(Image.open(item).convert("RGB"))
    return refs


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "/mnt/hf/models/flux2-dev-bnb4")
    device = os.environ.get("DEVICE", "cuda:0")
    out_path = Path(os.environ.get("OUT_PATH", "/out/flux2_dev_bnb4_256_2026-02-12.png"))
    prompt = os.environ.get(
        "PROMPT",
        "cinematic portrait photo of a person and robot sharing coffee in warm light",
    )
    steps = getenv_int("STEPS", 4)
    guidance = getenv_float("GUIDANCE", 3.0)
    height = getenv_int("HEIGHT", 256)
    width = getenv_int("WIDTH", 256)
    seed = getenv_int("SEED", 42)
    do_offload = os.environ.get("MODEL_CPU_OFFLOAD", "1") == "1"
    do_group_offload = getenv_bool("USE_GROUP_OFFLOAD", False)
    use_remote_text_encoder = os.environ.get("USE_REMOTE_TEXT_ENCODER", "1") == "1"
    max_sequence_length = getenv_int("MAX_SEQUENCE_LENGTH", 256)
    dtype = torch_dtype_from_env(os.environ.get("DTYPE", "bfloat16"))
    enable_vae_slicing = getenv_bool("VAE_SLICING", True)
    enable_vae_tiling = getenv_bool("VAE_TILING", False)
    group_offload_type = os.environ.get("GROUP_OFFLOAD_TYPE", "block_level")
    group_offload_blocks_raw = os.environ.get("GROUP_OFFLOAD_BLOCKS", "").strip()
    group_offload_blocks = int(group_offload_blocks_raw) if group_offload_blocks_raw else None
    group_offload_use_stream = getenv_bool("GROUP_OFFLOAD_USE_STREAM", False)
    callback_verbose = getenv_bool("CALLBACK_VERBOSE", True)
    init_image_path = os.environ.get("INIT_IMAGE", "").strip()
    text_encoder_retries = getenv_int("TEXT_ENCODER_RETRIES", 6)
    text_encoder_retry_sleep = getenv_int("TEXT_ENCODER_RETRY_SLEEP", 15)
    text_encoder_timeout = getenv_int("TEXT_ENCODER_TIMEOUT", 300)
    init_image: Image.Image | List[Image.Image] | None = None
    if init_image_path:
        refs = parse_image_refs(init_image_path)
        init_image = refs[0] if len(refs) == 1 else refs
        log(f"using {len(refs)} init image(s): {init_image_path}")

    log(f"loading pipeline from {model_id} ...")
    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if use_remote_text_encoder:
        load_kwargs["text_encoder"] = None

    pipe = Flux2Pipeline.from_pretrained(model_id, **load_kwargs)
    if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if do_group_offload and hasattr(pipe, "transformer") and hasattr(pipe.transformer, "enable_group_offload"):
        log(
            f"enabling group offload (type={group_offload_type}, blocks={group_offload_blocks}, "
            f"use_stream={group_offload_use_stream}) ..."
        )
        pipe.transformer.enable_group_offload(
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type=group_offload_type,
            num_blocks_per_group=group_offload_blocks,
            use_stream=group_offload_use_stream,
        )

    if do_offload and not do_group_offload:
        log("enabling model CPU offload ...")
        pipe.enable_model_cpu_offload()
    elif do_offload and do_group_offload:
        log("model CPU offload requested but group offload is active; skipping model CPU offload")
    else:
        pipe = pipe.to(device)

    log("running generation ...")
    generator_device = "cpu" if do_offload else device
    generate_kwargs = {
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "height": height,
        "width": width,
        "image": init_image,
        "max_sequence_length": max_sequence_length,
        "generator": torch.Generator(device=generator_device).manual_seed(seed),
    }
    if callback_verbose:
        def callback_on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            log(f"step {step_index + 1}/{steps}")
            return callback_kwargs

        generate_kwargs["callback_on_step_end"] = callback_on_step_end
    generate_kwargs = filter_supported_kwargs(pipe, generate_kwargs)

    if use_remote_text_encoder:
        log("requesting remote text encoder ...")
        response = None
        for attempt in range(1, text_encoder_retries + 1):
            try:
                response = requests.post(
                    "https://remote-text-encoder-flux-2.huggingface.co/predict",
                    json={"prompt": prompt},
                    headers={
                        "Authorization": f"Bearer {get_token()}",
                        "Content-Type": "application/json",
                    },
                    timeout=text_encoder_timeout,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                if attempt >= text_encoder_retries:
                    raise
                log(
                    f"remote text encoder attempt {attempt}/{text_encoder_retries} failed: {exc}; "
                    f"sleeping {text_encoder_retry_sleep}s"
                )
                time.sleep(text_encoder_retry_sleep)

        assert response is not None
        prompt_embeds = torch.load(io.BytesIO(response.content))
        if not do_offload:
            prompt_embeds = prompt_embeds.to(device)
        generate_kwargs["prompt_embeds"] = prompt_embeds
        image = pipe(**generate_kwargs).images[0]
    else:
        generate_kwargs["prompt"] = prompt
        image = pipe(**generate_kwargs).images[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    log(f"saved {out_path}")


if __name__ == "__main__":
    main()

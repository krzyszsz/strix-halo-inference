#!/usr/bin/env python3
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


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


def parse_image_refs(raw: str) -> list[Image.Image]:
    refs: list[Image.Image] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        refs.append(Image.open(item).convert("RGB"))
    return refs


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "/mnt/hf/models/flux2-klein-9b")
    out_path = Path(os.environ.get("OUT_PATH", "/out/flux2_klein_probe.png"))
    prompt = os.environ.get(
        "PROMPT",
        "A cinematic photo of a robot barista serving coffee in warm light",
    )
    height = getenv_int("HEIGHT", 512)
    width = getenv_int("WIDTH", 512)
    steps = getenv_int("STEPS", 4)
    guidance = getenv_float("GUIDANCE", 1.0)
    seed = getenv_int("SEED", 42)
    device = os.environ.get("DEVICE", "cuda:0")
    do_offload = os.environ.get("MODEL_CPU_OFFLOAD", "1") == "1"
    max_sequence_length = getenv_int("MAX_SEQUENCE_LENGTH", 512)
    text_encoder_out_layers_raw = os.environ.get("TEXT_ENCODER_OUT_LAYERS", "9,18,27")
    text_encoder_out_layers = tuple(
        int(part.strip()) for part in text_encoder_out_layers_raw.split(",") if part.strip()
    )
    init_image_raw = os.environ.get("INIT_IMAGE", "").strip()

    print(f"loading Flux2KleinPipeline from {model_id}", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if do_offload:
        print("enabling model CPU offload", flush=True)
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "max_sequence_length": max_sequence_length,
        "text_encoder_out_layers": text_encoder_out_layers,
    }

    if init_image_raw:
        refs = parse_image_refs(init_image_raw)
        kwargs["image"] = refs[0] if len(refs) == 1 else refs
        print(f"using {len(refs)} init image(s)", flush=True)

    print("running generation", flush=True)
    image = pipe(**kwargs).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

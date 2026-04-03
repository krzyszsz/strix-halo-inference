#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VGGT reconstruction and export a point cloud")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out-ply", required=True)
    p.add_argument("--out-preview", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--model-id", default="facebook/VGGT-1B")
    p.add_argument("--load-resolution", type=int, default=768)
    p.add_argument("--model-resolution", type=int, default=448)
    p.add_argument("--conf-threshold", type=float, default=0.0)
    p.add_argument("--max-points", type=int, default=200000)
    p.add_argument("--preview-points", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


def run_model(model: VGGT, images: torch.Tensor, model_resolution: int, dtype: torch.dtype, device: str):
    images = F.interpolate(images, size=(model_resolution, model_resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                images_batch = images[None]
                aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
        else:
            images_batch = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)

    return images, extrinsic.squeeze(0), intrinsic.squeeze(0), depth_map.squeeze(0), depth_conf.squeeze(0)


def make_preview(points: np.ndarray, colors: np.ndarray, out_preview: Path, preview_points: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    if n == 0:
        raise RuntimeError("No points left after filtering; cannot render preview")

    if n > preview_points:
        idx = rng.choice(n, size=preview_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    centered = points - np.mean(points, axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh[:2].T

    fig = plt.figure(figsize=(8, 8), dpi=220)
    ax = fig.add_subplot(111)
    ax.scatter(proj[:, 0], proj[:, 1], s=0.15, c=colors / 255.0, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    out_preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_preview, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return points.shape[0]


def main() -> None:
    args = parse_args()
    start = time.time()

    image_dir = Path(args.image_dir)
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least 2 input images, found {len(image_paths)} in {image_dir}")

    device, dtype = pick_device_and_dtype()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = VGGT.from_pretrained(args.model_id).to(device)
    model.eval()

    images, _ = load_and_preprocess_images_square([str(p) for p in image_paths], args.load_resolution)
    images = images.to(device)

    images_for_color, extrinsic, intrinsic, depth_map, depth_conf = run_model(
        model, images, args.model_resolution, dtype, device
    )

    points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    conf_np = depth_conf.detach().cpu().numpy() if isinstance(depth_conf, torch.Tensor) else np.asarray(depth_conf)
    if conf_np.ndim == 4 and conf_np.shape[-1] == 1:
        conf_np = conf_np[..., 0]

    colors_np = images_for_color.detach().cpu().numpy().transpose(0, 2, 3, 1)
    colors_np = np.clip(colors_np, 0.0, 1.0)

    finite_conf = np.isfinite(conf_np)
    fallback_conf_threshold = None
    if finite_conf.any():
        mask = finite_conf & (conf_np >= args.conf_threshold)
        if not mask.any():
            fallback_conf_threshold = float(np.nanpercentile(conf_np[finite_conf], 75.0))
            mask = finite_conf & (conf_np >= fallback_conf_threshold)
        if not mask.any():
            mask = finite_conf
            fallback_conf_threshold = None
    else:
        mask = np.ones(conf_np.shape, dtype=bool)
    points_flat = points[mask]
    colors_flat = (colors_np[mask] * 255.0).astype(np.uint8)

    finite_mask = np.isfinite(points_flat).all(axis=1)
    points_flat = points_flat[finite_mask]
    colors_flat = colors_flat[finite_mask]

    rng = np.random.default_rng(args.seed)
    if points_flat.shape[0] > args.max_points:
        idx = rng.choice(points_flat.shape[0], size=args.max_points, replace=False)
        points_flat = points_flat[idx]
        colors_flat = colors_flat[idx]

    if points_flat.shape[0] == 0:
        raise RuntimeError("No valid points generated after filtering")

    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(points_flat, colors=colors_flat).export(out_ply)

    out_preview = Path(args.out_preview)
    preview_points = make_preview(points_flat, colors_flat, out_preview, args.preview_points, args.seed)

    elapsed = time.time() - start

    summary = {
        "model_id": args.model_id,
        "device": device,
        "dtype": str(dtype),
        "input_image_count": len(image_paths),
        "input_images": [rel_for_report(p) for p in image_paths],
        "load_resolution": args.load_resolution,
        "model_resolution": args.model_resolution,
        "confidence_threshold": args.conf_threshold,
        "confidence_min": float(np.nanmin(conf_np)),
        "confidence_max": float(np.nanmax(conf_np)),
        "confidence_mean": float(np.nanmean(conf_np)),
        "confidence_fallback_threshold": fallback_conf_threshold,
        "max_points": args.max_points,
        "num_points_written": int(points_flat.shape[0]),
        "preview_points": int(preview_points),
        "out_ply": rel_for_report(out_ply),
        "out_preview": rel_for_report(out_preview),
        "elapsed_seconds": round(elapsed, 2),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(out_json)
    print(out_ply)
    print(out_preview)


if __name__ == "__main__":
    main()

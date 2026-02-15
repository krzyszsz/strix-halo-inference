#!/usr/bin/env python3
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass(frozen=True)
class Detection:
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2 in pixels
    score: float


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


def list_images(folder: Path) -> List[Path]:
    out: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        out.extend(sorted(folder.glob(ext)))
    return sorted(set(out))


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return float(inter) / float(area_a + area_b - inter)


def nms(dets: List[Detection], iou_threshold: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for d in dets:
        if all(iou(d.box, k.box) < iou_threshold for k in kept):
            kept.append(d)
    return kept


def make_prior_boxes(input_w: int, input_h: int) -> np.ndarray:
    # UltraFace priors for RFB-320 (matches official example code).
    # min_boxes correspond to feature map layers.
    min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
    strides = [8.0, 16.0, 32.0, 64.0]

    priors: List[List[float]] = []
    for stride, boxes in zip(strides, min_boxes, strict=True):
        fm_w = int(math.ceil(input_w / stride))
        fm_h = int(math.ceil(input_h / stride))
        for y in range(fm_h):
            for x in range(fm_w):
                cx = (x + 0.5) * stride / input_w
                cy = (y + 0.5) * stride / input_h
                for box in boxes:
                    w = box / input_w
                    h = box / input_h
                    priors.append([cx, cy, w, h])
    return np.asarray(priors, dtype=np.float32)


def decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    # loc shape: [N, 4] (cx, cy, w, h deltas)
    center_variance = 0.1
    size_variance = 0.2
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * center_variance * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * size_variance),
        ),
        axis=1,
    )
    # cx,cy,w,h -> x1,y1,x2,y2
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class UltraFaceDetector:
    def __init__(self, model_path: Path, threads: int = 8) -> None:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = threads
        sess_opts.inter_op_num_threads = max(1, threads // 2)
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.in_h, self.in_w = self.session.get_inputs()[0].shape
        self.priors = make_prior_boxes(self.in_w, self.in_h)

    def detect(self, image_rgb: np.ndarray, conf_threshold: float, iou_threshold: float, top_k: int) -> List[Detection]:
        h, w, _ = image_rgb.shape
        resized = cv2.resize(image_rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        # Official preprocessing: mean 127, scale 1/128, RGB order, NCHW
        blob = resized.astype(np.float32)
        blob = (blob - 127.0) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

        scores, boxes = self.session.run(None, {self.input_name: blob})
        scores = scores[0]  # [N,2]
        boxes = boxes[0]  # [N,4]
        # Score is class 1 ("face")
        face_scores = scores[:, 1]
        mask = face_scores > conf_threshold
        if not np.any(mask):
            return []

        filtered_scores = face_scores[mask]
        filtered_locs = boxes[mask]
        filtered_priors = self.priors[mask]

        decoded = decode_boxes(filtered_locs, filtered_priors)
        # to pixel coords on resized input
        decoded[:, 0] *= w
        decoded[:, 1] *= h
        decoded[:, 2] *= w
        decoded[:, 3] *= h

        dets = []
        for (x1, y1, x2, y2), score in zip(decoded, filtered_scores, strict=True):
            dets.append(
                Detection(
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(score),
                )
            )

        dets = nms(dets, iou_threshold)
        return dets[:top_k]


class ArcFaceRecognizer:
    def __init__(self, model_path: Path, threads: int = 8) -> None:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = threads
        sess_opts.inter_op_num_threads = max(1, threads // 2)
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.in_h, self.in_w = self.session.get_inputs()[0].shape

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32)
        # NOTE: The ONNX Model Zoo ArcFace export we use (`arcfaceresnet100-8`) expects
        # raw 0-255 pixel values (float). Applying the usual "(x-127.5)/128" here
        # collapses embeddings (unrelated inputs become near-identical).
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]
        out = self.session.run(None, {self.input_name: blob})[0]
        emb = out[0].astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-12
        return emb / norm


def crop_with_margin(img_rgb: np.ndarray, box: Tuple[int, int, int, int], margin: float) -> np.ndarray:
    h, w, _ = img_rgb.shape
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(w - 1, x2 + mx)
    ny2 = min(h - 1, y2 + my)
    return img_rgb[ny1 : ny2 + 1, nx1 : nx2 + 1, :]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def make_synthetic_collage(ref_dir: Path, out_group: Path, out_expected: Path) -> None:
    # Build a deterministic 2x2 collage from existing refs, with small augmentations.
    refs = list_images(ref_dir)
    if len(refs) < 4:
        raise RuntimeError(f"Need at least 4 reference images in {ref_dir}")

    chosen = refs[:4]
    labels = [p.stem for p in chosen]
    imgs: List[Image.Image] = []
    for idx, path in enumerate(chosen):
        im = Image.open(path).convert("RGB")
        # Keep this a simple, high-signal smoke test: no strong augmentation.
        # (Alignment is not implemented in this minimal demo.)
        im = im.resize((512, 512), Image.Resampling.LANCZOS)
        imgs.append(im)

    cols = 2
    rows = 2
    pad = 24
    cell_w, cell_h = 512, 512
    out_w = cols * cell_w + (cols + 1) * pad
    out_h = rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), (20, 24, 28))

    expected_cells = []
    for i, (label, im) in enumerate(zip(labels, imgs, strict=True)):
        r = i // cols
        c = i % cols
        x1 = pad + c * (cell_w + pad)
        y1 = pad + r * (cell_h + pad)
        canvas.paste(im, (x1, y1))
        expected_cells.append(
            {
                "label": label,
                "rect": [x1, y1, x1 + cell_w, y1 + cell_h],
            }
        )

    out_group.parent.mkdir(parents=True, exist_ok=True)
    out_expected.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_group)
    out_expected.write_text(json.dumps({"cells": expected_cells}, indent=2), encoding="utf-8")


def find_expected_label(expected: dict, center_x: int, center_y: int) -> Optional[str]:
    for cell in expected.get("cells", []):
        x1, y1, x2, y2 = cell.get("rect", [0, 0, 0, 0])
        if x1 <= center_x < x2 and y1 <= center_y < y2:
            return str(cell.get("label"))
    return None


def main() -> None:
    detector_model = Path(os.environ.get("DETECTOR_MODEL", "/models/ultraface/version-RFB-320.onnx"))
    recognizer_model = Path(os.environ.get("RECOGNIZER_MODEL", "/models/arcface/arcfaceresnet100-8.onnx"))
    ref_dir = Path(os.environ.get("REF_DIR", "/input/ref"))
    query_image_path = Path(os.environ.get("QUERY_IMAGE", "/input/group.png"))
    expected_path = Path(os.environ.get("EXPECTED_JSON", "/input/group_expected.json"))
    out_json = Path(os.environ.get("OUT_JSON", "/out/face_match_results.json"))
    out_image = Path(os.environ.get("OUT_IMAGE", "/out/face_match_annotated.png"))
    out_summary = Path(os.environ.get("OUT_SUMMARY", "/out/face_match_summary.json"))

    threads = getenv_int("THREADS", 8)
    conf_threshold = getenv_float("CONF_THRESHOLD", 0.7)
    iou_threshold = getenv_float("IOU_THRESHOLD", 0.3)
    top_k = getenv_int("TOP_K", 50)
    margin = getenv_float("CROP_MARGIN", 0.25)

    if os.environ.get("MAKE_SYNTH_COLLAGE", "1") == "1":
        log("building synthetic collage from refs ...")
        make_synthetic_collage(ref_dir, query_image_path, expected_path)

    log(f"loading detector: {detector_model}")
    detector = UltraFaceDetector(detector_model, threads=threads)
    log(f"loading recognizer: {recognizer_model}")
    recognizer = ArcFaceRecognizer(recognizer_model, threads=threads)

    ref_images = list_images(ref_dir)
    if not ref_images:
        raise SystemExit(f"no reference images found in {ref_dir}")

    ref_embeddings: Dict[str, np.ndarray] = {}
    for path in ref_images:
        label = path.stem
        rgb = np.array(Image.open(path).convert("RGB"))
        dets = detector.detect(rgb, conf_threshold=conf_threshold, iou_threshold=iou_threshold, top_k=top_k)
        if not dets:
            log(f"ref {path.name}: no face detected; skipping")
            continue
        # pick best score (usually single face).
        det = max(dets, key=lambda d: d.score)
        face = crop_with_margin(rgb, det.box, margin=margin)
        ref_embeddings[label] = recognizer.embed(face)
        log(f"ref {path.name}: detected score={det.score:.3f}")

    if not ref_embeddings:
        raise SystemExit("no reference embeddings computed (all detections failed)")

    query_rgb = np.array(Image.open(query_image_path).convert("RGB"))
    query_dets = detector.detect(query_rgb, conf_threshold=conf_threshold, iou_threshold=iou_threshold, top_k=top_k)
    log(f"query detections: {len(query_dets)} face(s)")

    expected: dict = {}
    if expected_path.exists():
        try:
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
        except Exception:
            expected = {}

    results = {
        "models": {
            "detector": str(detector_model),
            "recognizer": str(recognizer_model),
        },
        "thresholds": {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "crop_margin": margin,
        },
        "references": [{"label": k, "file": f"{k}{Path(ref_images[0]).suffix}"} for k in sorted(ref_embeddings.keys())],
        "query": {
            "file": query_image_path.name,
            "detections": [],
        },
    }

    correct = 0
    evaluated = 0
    best_by_cell: Dict[str, dict] = {}

    # annotate
    anno = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)

    for det in query_dets:
        face = crop_with_margin(query_rgb, det.box, margin=margin)
        emb = recognizer.embed(face)

        sims = {label: cosine_similarity(emb, ref_emb) for label, ref_emb in ref_embeddings.items()}
        best_label = max(sims, key=sims.get)
        best_sim = sims[best_label]

        cx = (det.box[0] + det.box[2]) // 2
        cy = (det.box[1] + det.box[3]) // 2
        expected_label = find_expected_label(expected, cx, cy) if expected else None
        is_correct = expected_label is not None and expected_label == best_label
        if expected_label is not None:
            # Track best detection per expected cell so duplicates don't tank the score.
            current = best_by_cell.get(expected_label)
            candidate = {
                "det_score": det.score,
                "best_label": best_label,
                "best_cosine": best_sim,
                "correct": is_correct,
            }
            if current is None or det.score > float(current.get("det_score", 0.0)):
                best_by_cell[expected_label] = candidate

        results["query"]["detections"].append(
            {
                "box_xyxy": list(det.box),
                "score": det.score,
                "center_xy": [cx, cy],
                "match": {
                    "best_label": best_label,
                    "best_cosine": best_sim,
                    "expected_label": expected_label,
                    "correct": is_correct if expected_label is not None else None,
                },
                "cosine_by_label": {k: float(v) for k, v in sorted(sims.items(), key=lambda kv: kv[1], reverse=True)},
            }
        )

        x1, y1, x2, y2 = det.box
        cv2.rectangle(anno, (x1, y1), (x2, y2), (0, 220, 0), 2)
        text = f"{best_label} {best_sim:.3f}"
        cv2.putText(anno, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    cv2.imwrite(str(out_image), anno)

    if expected and expected.get("cells"):
        evaluated = 0
        correct = 0
        for cell in expected.get("cells", []):
            label = str(cell.get("label"))
            best = best_by_cell.get(label)
            if best is None:
                continue
            evaluated += 1
            correct += 1 if bool(best.get("correct")) else 0

    summary = {
        "reference_count": len(ref_embeddings),
        "detected_faces": len(query_dets),
        "evaluated_faces": evaluated,
        "correct_matches": correct,
        "hit_rate": (correct / evaluated) if evaluated else None,
        "scoring": "per_expected_cell_best_det" if expected and expected.get("cells") else "per_detection",
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"saved {out_json}")
    log(f"saved {out_image}")
    log(f"saved {out_summary}")
    if summary["hit_rate"] is not None:
        log(f"hit_rate={summary['hit_rate']:.3f} ({correct}/{evaluated})")


if __name__ == "__main__":
    main()

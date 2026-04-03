import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
from ultralytics import YOLO


def _to_float_list(values) -> List[float]:
    return [float(v) for v in values]


def _rel_path(path: str) -> str:
    p = Path(path)
    repo_root = os.environ.get("REPO_ROOT")
    candidates = []
    if repo_root:
        candidates.append(Path(repo_root).resolve())
    candidates.append(Path.cwd().resolve())
    for base in candidates:
        try:
            return str(p.resolve().relative_to(base))
        except Exception:
            continue
    return str(p)


def run_detection(model_path: str, image_path: str, device: str, conf: float, imgsz: int) -> Any:
    model = YOLO(model_path)
    results = model.predict(source=image_path, device=device, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        raise RuntimeError("No results returned by YOLO")
    return model, results[0]


def serialize_boxes(result, class_names: Dict[int, str]) -> List[Dict[str, Any]]:
    boxes = []
    if result.boxes is None:
        return boxes
    for b in result.boxes:
        cls_id = int(b.cls.item())
        boxes.append(
            {
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, str(cls_id)),
                "confidence": float(b.conf.item()),
                "xyxy": _to_float_list(b.xyxy[0].tolist()),
            }
        )
    return boxes


def serialize_keypoints(result) -> List[Dict[str, Any]]:
    kps = []
    if result.keypoints is None:
        return kps
    xy = result.keypoints.xy
    conf = result.keypoints.conf
    for idx in range(len(xy)):
        keypoints = []
        for kp_i in range(xy[idx].shape[0]):
            keypoint = {
                "x": float(xy[idx][kp_i][0].item()),
                "y": float(xy[idx][kp_i][1].item()),
            }
            if conf is not None:
                keypoint["confidence"] = float(conf[idx][kp_i].item())
            keypoints.append(keypoint)
        kps.append({"person_index": idx, "keypoints": keypoints})
    return kps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detect/pose and export coordinates")
    parser.add_argument("--task", choices=["detect", "pose"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-image", required=True)
    args = parser.parse_args()

    model, result = run_detection(
        model_path=args.model,
        image_path=args.input,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    boxes = serialize_boxes(result, model.names)
    payload: Dict[str, Any] = {
        "task": args.task,
        "model": args.model,
        "input": _rel_path(args.input),
        "device": args.device,
        "confidence_threshold": args.conf,
        "image_size": args.imgsz,
        "num_detections": len(boxes),
        "boxes": boxes,
    }

    if args.task == "pose":
        payload["poses"] = serialize_keypoints(result)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    plotted = result.plot()
    out_image = Path(args.out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), plotted)

    print(out_json)
    print(out_image)


if __name__ == "__main__":
    main()

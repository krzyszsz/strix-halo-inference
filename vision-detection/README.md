# Vision Detection and Pose (Coordinates)

This folder documents a local coordinate-extraction workflow for:
- item/person detection (bounding boxes), and
- human pose keypoints (17-point skeleton).

It uses Ultralytics YOLO models in a reproducible Docker container and follows the same memory-safe policy as the rest of this repository.

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

## Build

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/vision-detection/scripts/build_vision_yolo.sh
```

## Run detection (item/person bounding boxes)

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_NAME=yolo26n.pt \
      DEVICE=cpu \
      INPUT_IMAGE=$REPO_ROOT/vision-detection/input/bus.jpg \
      OUT_JSON=$REPO_ROOT/vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.json \
      OUT_IMAGE=$REPO_ROOT/vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg \
  bash $REPO_ROOT/vision-detection/scripts/test_yolo_detect.sh
```

Output JSON includes class name, confidence, and `xyxy` coordinates.

## Run pose (human keypoints)

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_NAME=yolo26n-pose.pt \
      DEVICE=cpu \
      INPUT_IMAGE=$REPO_ROOT/vision-detection/input/bus.jpg \
      OUT_JSON=$REPO_ROOT/vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.json \
      OUT_IMAGE=$REPO_ROOT/vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg \
  bash $REPO_ROOT/vision-detection/scripts/test_yolo_pose.sh
```

Output JSON includes person boxes and per-person keypoints (`x`, `y`, `confidence`) suitable as inputs for inverse-kinematics-like downstream logic.

## Evidence from this repo

- Build log:
  - `reports/publish/build_vision_yolo.log`
- Detection logs/artifacts:
  - `reports/publish/yolo26n_detect_bus.log`
  - `vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.json`
  - `vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg`
- Pose logs/artifacts:
  - `reports/publish/yolo26n_pose_bus.log`
  - `vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.json`
  - `vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg`

## Model/license references

- YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
- Detection task docs: https://docs.ultralytics.com/tasks/detect/
- Pose task docs: https://docs.ultralytics.com/tasks/pose/
- Ultralytics license: https://github.com/ultralytics/ultralytics/blob/main/LICENSE

# Face Detection + Recognition (CPU, ONNXRuntime)

This folder documents a local, CPU-only face detection + embedding + matching workflow.

Important:
- Public docs in this repository do **not** include biometric identification of real people.
- The demo uses synthetic/AI-generated portraits labeled as `person_1`, `person_2`, etc.

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

## What This Demo Does

1. Uses a face detector (UltraFace) to find face boxes.
2. Uses a face embedding model (ArcFace) to embed each detected face.
3. Matches each query face to the closest reference embedding (cosine similarity).

To keep this safe to publish, the reference set is generated from locally-produced synthetic portraits and the demo reports only `person_N` labels.

## Build

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/vision-face/scripts/build_vision_face.sh
```

## Download models (to `$MODEL_ROOT`)

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/vision-face/scripts/download_face_models.sh
```

Downloads:
- Face detector (UltraFace): `onnxmodelzoo/version-RFB-320`
- Face embedding model (ArcFace): `onnxmodelzoo/arcfaceresnet100-8`

## Run the demo (generate synthetic refs + group collage + match)

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/vision-face/scripts/test_face_match.sh
```

Outputs:
- `vision-face/out/face_match_results.json`
- `vision-face/out/face_match_annotated.png`
- `vision-face/out/face_match_summary.json`

Result (latest):
- `hit_rate=0.75` (`3/4`) on the synthetic 2x2 collage (duplicates are ignored via per-cell best-detection scoring).

![Face match annotated](out/face_match_annotated.png)
Caption: `UltraFace + ArcFace` | synthetic collage | CPU-only | evidence: `reports/publish/vision_face_match_demo.log`

Evidence (this repo):
- `reports/publish/vision_face_match_demo.log`
- `vision-face/out/face_match_annotated.png`
- `vision-face/out/face_match_results.json`
- `vision-face/out/face_match_summary.json`

Notes:
- For this ArcFace ONNX export, embeddings only behave correctly if the recognizer is fed **raw 0-255 float pixels**. Applying the common `(x-127.5)/128` normalization collapses embeddings and breaks matching.

# 3D Reconstruction From Photos (VGGT)

This folder contains a reproducible 3D reconstruction workflow from a multi-view photo set.

Model used:
- `facebook/VGGT-1B` (CVPR 2025), a feed-forward geometry model that predicts cameras + depth from image sets.

Input set for this repo:
- 12 contiguous views from COLMAP's `south-building` sample sequence (same camera/date/light).

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

## Build image

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/reconstruction-3d/scripts/build_vggt_image.sh
```

## Download photo set

```bash
bash $REPO_ROOT/reconstruction-3d/scripts/download_south_building_views.sh
```

## Run reconstruction

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/reconstruction-3d/scripts/run_vggt_reconstruct.sh
```

Outputs:
- `reconstruction-3d/out/south_building/south_building_points.ply`
- `reconstruction-3d/out/south_building/south_building_points_preview.png`
- `reconstruction-3d/out/south_building/south_building_summary.json`

## JavaScript viewer

```bash
PORT=8093 bash $REPO_ROOT/reconstruction-3d/scripts/serve_viewer.sh
```

Open:
- `http://127.0.0.1:8093/reconstruction-3d/viewer/`

Viewer stack:
- `three.js` + `PLYLoader` + `OrbitControls`

## References

- VGGT project: https://github.com/facebookresearch/vggt
- VGGT paper: https://arxiv.org/abs/2503.11651
- VGGT model card: https://huggingface.co/facebook/VGGT-1B
- COLMAP sample datasets: https://github.com/colmap/colmap/releases

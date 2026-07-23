# RuView baseline

Comparison against [ruvnet/RuView](https://github.com/ruvnet/RuView)
(wifi-densepose), the most visible open-source WiFi-pose project. Their repo is a
git submodule at `RuView/`; we import their code directly:

- `RuView/examples/through-wall/wiflow_train.py`: their CSI->keypoints MLP (`Net`)
- `RuView/archive/v1/src/models/densepose_head.py`: their DensePose head and
  segmentation loss

`run_skeleton.py` / `run_densepose.py` train these on our session A with their
input representation (windowed raw-amplitude statistics, no Doppler) and
evaluate on the same separate-recording holdout and metrics as the main repo.

## Results (holdout)

| Metric | RuView (raw amplitude) | wifipose (Doppler) |
|---|---|---|
| Skeleton MPJPE | 85.0 mm (constant pose: 79.5) | 53.9 mm |
| Skeleton wrist-z r | -0.06 | +0.22 |
| DensePose fg-IoU | 0.120 | 0.260 |
| DensePose part-mIoU | 0.007 (24-part) | 0.070 (5-part) |

The skeleton model is worse than predicting the training-mean pose and shows
zero wrist tracking. The segmentation head localizes a foreground blob but
cannot separate body parts. One nuance in their favor: its per-frame
matched-vs-shuffled fg-IoU gap (0.043) is larger than ours (~0.005) — raw
amplitude modulates the foreground frame to frame more than our Doppler
model does, it just puts the parts in the wrong places.

Notes on the upstream project itself: its DSP pipeline (`verify.py`) passes
bit-exact, but the shipped pretrained pose model (`pose_v1.safetensors`) is a
constant predictor (output std 0.0015; PCK@20 2.97% under its own eval), and
the advertised MM-Fi numbers come from a separate multi-antenna variant, not
the shipped model. Consistent with our result: their architecture is fine, the
raw-amplitude single-antenna input is the failure point.

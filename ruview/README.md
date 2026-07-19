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
| Skeleton MPJPE | 85.0 mm (constant pose: 79.5) | 54.3 mm |
| Skeleton wrist-z r | -0.06 | +0.21 |
| DensePose fg-IoU | 0.147 | 0.167 |
| DensePose part-mIoU | 0.009 (24-part) | 0.051 (5-part) |

The skeleton model is worse than predicting the training-mean pose and shows
zero wrist tracking. The segmentation head localizes a foreground blob but
cannot separate body parts.

Notes on the upstream project itself: its DSP pipeline (`verify.py`) passes
bit-exact, but the shipped pretrained pose model (`pose_v1.safetensors`) is a
constant predictor (output std 0.0015; PCK@20 2.97% under its own eval), and
the advertised MM-Fi numbers come from a separate multi-antenna variant, not
the shipped model. Consistent with our result: their architecture is fine, the
raw-amplitude single-antenna input is the failure point.

# RuView baseline

Comparison against [ruvnet/RuView](https://github.com/ruvnet/RuView)
(wifi-densepose), the most visible open-source WiFi-pose project. `RuView/` is
a git submodule pointing at that repository, and the runners here import their
code directly:

- `RuView/examples/through-wall/wiflow_train.py`: their CSI to keypoints MLP (`Net`)
- `RuView/archive/v1/src/models/densepose_head.py`: their DensePose head and
segmentation loss

`run_skeleton.py` and `run_densepose.py` train those on our training session
using their input representation (windowed raw-amplitude statistics, no
Doppler), and evaluate on the same separate-recording holdout with the same
metrics as the main repo.

## Results (holdout)

| Metric | RuView (raw amplitude) | wifipose (Doppler) |
|---|---|---|
| Skeleton MPJPE | 85.0 mm (constant pose: 79.5) | 53.9 mm |
| Skeleton wrist-z r | -0.06 | +0.22 |
| DensePose fg-IoU | 0.120 | 0.260 |
| DensePose part-mIoU | 0.007 (24-part) | 0.070 (5-part) |

Their skeleton model is worse than predicting the training-mean pose and shows
no wrist tracking. Their segmentation head localizes a foreground blob but does
not separate body parts.

One nuance in their favor: their per-frame matched-versus-shuffled fg-IoU gap
is 0.043, where ours is 0.0016. Raw amplitude modulates the predicted
foreground frame to frame considerably more than our Doppler model does. It
just puts the parts in the wrong places.

## Notes on the upstream project

Their DSP pipeline (`verify.py`) passes bit-exact, so the signal processing is
real and deterministic. The shipped pretrained pose model
(`pose_v1.safetensors`) is a constant predictor: output standard deviation
0.0015 across 500 varied inputs, and PCK@20 of 2.97% under their own
evaluation. Their published 82.7% torso-PCK@20 comes from a separate
multi-antenna MM-Fi variant, not the shipped single-node model.

This is consistent with our own result. Their architecture is reasonable; the
raw-amplitude single-antenna input is the failure point.

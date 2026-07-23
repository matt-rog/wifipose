# wifipose

Pose estimation from single-antenna WiFi CSI. A Raspberry Pi 3B+ (nexmon, 80 MHz)
captures CSI from a WiFi router's transmissions and forwards it to a laptop,
where a synced camera records the subject. CoMotion and
DensePose run on the video to produce labels. Doppler features are extracted from
the CSI amplitude and small models are trained to predict pose, 2D keypoints, and
coarse body-part maps. Evaluation is on a separate recording, against a
constant-pose baseline, with wrist correlation and empty-room checks.

## Results (separate-recording holdout)

| Output | Result | Baseline |
|---|---|---|
| Arm elevation (1-D) | r 0.39 | — |
| SMPL-24 skeleton | 54 mm MPJPE, wrist-z r 0.22 | constant pose: 80 mm |
| 2D keypoints | PCK@0.2 0.73 | constant pose: 0.56 |
| Body-part maps (5-class) | fg-IoU 0.26, empty-room fg 3% | static blob: 0.09 |

Only arm motion is tracked; static joints come from the pose prior
(see `arms_only_mpjpe` in the skeleton report). `ruview/` runs the RuView
baseline on the same data for comparison.

## Contents

```
record/    pi bringup, synchronized camera + CSI recorder, sync calibration
teacher/   CoMotion SMPL targets, DensePose ground truth
wifipose/  CSI loading, Doppler features, metrics
train/     wave, skeleton, 2D keypoint, and body-part trainers
infer/     holdout video renderer
ruview/    RuView baseline (submodule + runners)
```

## Reproduce

```bash
./record/pi_bringup.sh <pi_ip>
python record/record.py --prefix sync --secs 30      # jumping jacks for clock sync
python record/record.py --prefix train --secs 840    # training session
python record/record.py --prefix empty --secs 150    # empty room, nobody present
python record/record.py --prefix holdout --secs 36   # separate recording for eval
python record/sync_offset.py --prefix sync --mac <bssid>

./teacher/run_comotion.sh train_video.avi <n_frames> train.pt
python teacher/comotion_targets.py --pt train.pt --frame-ts train_frame_ts.npy \
    --sync-offset <offset> --out train_Y.npz          # repeat for holdout
python teacher/densepose_gt.py --video train_video.avi --out train_dp.npz

python train/train_wave.py --mac <bssid>
python train/train_skeleton.py --mac <bssid>
python train/train_openpose.py --mac <bssid>
python train/train_densepose.py --mac <bssid>

python infer/render_holdout.py --mac <bssid>
```

Requires `numpy opencv-python torch`, plus apple/ml-comotion and
detectron2+DensePose for label generation. Clone with `--recursive`.

## References

- [nexmon_csi](https://github.com/seemoo-lab/nexmon_csi), [picsi](https://github.com/nexmonster/picsi)
- [CARM (MobiCom 2015)](https://www.sigmobile.org/mobicom/2015/papers/p65-wangA.pdf)
- [SHARP (IEEE TMC 2023)](https://arxiv.org/abs/2103.09924)
- [Widar3.0 (MobiSys 2019)](https://tns.thss.tsinghua.edu.cn/widar3.0/)
- [WiPose (MobiCom 2020)](https://cse.buffalo.edu/~lusu/papers/MobiCom2020.pdf)
- [CoMotion](https://github.com/apple/ml-comotion)
- [DensePose / detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
- [DensePose From WiFi](https://arxiv.org/abs/2301.00250)
- [MM-Fi (NeurIPS 2023)](https://arxiv.org/abs/2305.10345)
- [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
- [Strohmayer & Kampel (augmentation)](https://arxiv.org/abs/2401.00964)
- [RuView](https://github.com/ruvnet/RuView)

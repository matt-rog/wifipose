# wifipose

Human pose estimation from single-antenna WiFi CSI. One Raspberry Pi 3B+
(nexmon CSI, 80 MHz), one commodity AP, one synchronized camera as the label
source. No wearables, no multi-antenna array.

Trained on one ~14 min recording of one subject; evaluated on a separate
recording the model never saw.

| Output | Holdout result | Baseline |
|---|---|---|
| Arm elevation (1-D) | Pearson r 0.39 | — |
| SMPL-24 skeleton | MPJPE 54 mm, PA-MPJPE 47 mm, wrist-z r 0.21 | constant pose: 80 mm, r 0 |
| Coarse body parts (5-class, 36x64) | fg-IoU 0.17, empty-room fg 1% | static blob: 0.09 |

The RuView baseline (`ruview/`, their model code on the same data)
scores below the constant-pose baseline on skeletons: raw amplitude does not
transfer across recordings, Doppler does.

## How it works

```
Pi (picsi/nexmon) --UDP--> record/record.py <-- camera        one monotonic clock
                                |
              teacher: CoMotion (SMPL-24) + detectron2 DensePose on the video
                                |
        wifipose/dfs.py: Doppler features from CSI amplitude (2s windows)
                                |
              train/: small MLP / FCN heads, seed ensembles
```

The core representation is the Doppler spectrum of the motion-induced CSI
amplitude component. Raw CSI amplitude encodes environment-specific static
multipath and does not transfer across recordings; the Doppler component
depends on body kinematics and does (measured here: wrist correlation 0.07 ->
0.21 from this change alone). Single-antenna phase is discarded: per-packet
CFO/SFO offsets make it unusable without a second co-oscillator RX chain.

## Prior work this builds on

| Component | Source |
|---|---|
| CSI firmware + Pi tooling | [seemoo-lab/nexmon_csi](https://github.com/seemoo-lab/nexmon_csi), [nexmonster/picsi](https://github.com/nexmonster/picsi) |
| Doppler-from-amplitude principle | CARM (MobiCom 2015); SHARP (IEEE TMC 2023, [code](https://github.com/signetlabdei/SHARP)) — single-environment, single-person training viability |
| Multi-link Doppler as the domain-independent feature | Widar3.0 (MobiSys 2019) — this repo uses its single-link slice |
| Pose teacher | [apple/ml-comotion](https://github.com/apple/ml-comotion) (CoMotion, SMPL-24) |
| Part-map teacher | [detectron2 DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) |
| Skeleton priors (bone-length + smoothness losses) | WiPose (MobiCom 2020) |
| Time-shift / frequency-mask augmentation | Strohmayer & Kampel (arXiv 2401.00964, single-antenna cross-domain ablation); RadarSpecAugment (IEEE Sensors Letters 2021) |
| Metrics (MPJPE / PA-MPJPE / torso-PCK) | MM-Fi benchmark (NeurIPS 2023 D&B); model scale follows SenseFi ([code](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)) |
| Coarse-target choice | DensePose From WiFi (arXiv 2301.00250) needs 3x3 antennas for fine parts; 1x1 supports only coarse maps |

## Evaluation rules

- The holdout is a **separate recording**. Random or within-session splits
  overestimate WiFi sensing performance badly (documented across the field;
  reproduced here).
- Always report the **constant-pose baseline**: on a mostly-static subject the
  training-mean pose alone reaches ~80% of MPJPE.
- **Wrist correlation** and the **prediction-variance ratio** are the honesty
  metrics: a mean-collapsed model can score well on MPJPE/PCK but cannot fake
  either.
- Segmentation reports the **matched vs time-shuffled** per-frame fg-IoU gap
  (a static person-blob scores the same on both) and the **empty-room
  foreground fraction** (person-hallucination check; empty-room windows are
  also training negatives).
- The within-session val tail is used for early stopping only — it cannot rank
  cross-recording transfer (measured: models that win val can lose holdout).

## Reproduce

```bash
# 0. hardware: Pi with picsi installed, AP on an 80 MHz channel, USB camera
./record/pi_bringup.sh <pi_ip>                     # CSI -> laptop UDP relay

# 1. record (sync take + session + empty room + a separate holdout take)
python record/record.py --prefix sync --secs 30    # jumping jacks
python record/record.py --prefix A    --secs 840   # training session
python record/record.py --prefix A_empty --secs 150  # nobody in the room
python record/record.py --prefix demo --secs 36    # holdout
python record/sync_offset.py --prefix sync --mac <bssid>

# 2. teacher labels (GPU; needs apple/ml-comotion and detectron2+DensePose)
./teacher/run_comotion.sh A_video.avi <n_frames> A.pt
python teacher/comotion_targets.py --pt A.pt --frame-ts A_frame_ts.npy \
    --sync-offset <offset> --out A_Y.npz            # same for demo
python teacher/densepose_gt.py --video A_video.avi --out A_dp.npz   # same for demo

# 3. train + evaluate on the holdout
python train/train_wave.py     --train A --holdout demo --mac <bssid>
python train/train_skeleton.py --train A --holdout demo --mac <bssid>
python train/train_densepose.py --train A --holdout demo --empty A_empty --mac <bssid>

# 4. videos
python infer/render_holdout.py --holdout demo --mac <bssid>
```

## Limits

One antenna gives a 1-D radial Doppler projection: coarse motion tracking, not
fine skeletons or per-pixel parts. Every published system with strong
cross-domain pose (Widar3.0-class) uses >=3 RX links to build body-coordinate
velocity features. The next hardware step is a second RX antenna (CSI-ratio
phase, FarSense IMWUT 2019), which unlocks the signed Doppler and AoA this
setup cannot measure.

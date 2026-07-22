#!/usr/bin/env python3
"""Render holdout predictions next to the camera and teacher ground truth.

python infer/render_holdout.py --holdout demo --mac <bssid> --data data
"""
import argparse, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wifipose.csi import load_csi
from wifipose.dfs import valid_mask
from wifipose.metrics import PARENTS, PELV

BONES = [(c, PARENTS[c]) for c in range(1, 24)]

# OpenPose COCO-18 rendering constants (CMU openpose poseParametersRender.hpp)
OP_PAIRS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
            (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
            (0, 15), (15, 17)]
OP_COLORS = [(255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
             (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
             (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255),
             (0, 85, 255), (0, 0, 255), (255, 0, 170), (170, 0, 255),
             (255, 0, 255), (85, 0, 255)]
# COCO-17 (detectron2) index for each COCO-18 joint; neck (-1) = mid-shoulders
COCO18_FROM_17 = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

# DensePose official visualizer recipe (detectron2 FineSegmentationVisualizer)
DP_CMAP, DP_ALPHA = cv2.COLORMAP_PARULA, 0.7
COARSE = np.zeros(25, np.int64)
COARSE[[1, 2]] = 1
COARSE[[23, 24]] = 2
COARSE[[3, 4, 15, 16, 17, 18, 19, 20, 21, 22]] = 3
COARSE[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] = 4


def to_coco18(kp17):
    kp = np.array([kp17[i] if i >= 0 else 0.5 * (kp17[5] + kp17[6])
                   for i in COCO18_FROM_17])
    return kp


def draw_openpose(frame, kp18, scale, thin=False):
    """CMU openpose COCO body rendering: per-limb colors, ellipse limbs,
    alpha-blended onto the frame."""
    canvas = frame.copy()
    pts = (kp18 * scale).astype(int)
    for (i, j), color in zip(OP_PAIRS, OP_COLORS):
        (x1, y1), (x2, y2) = pts[i], pts[j]
        if thin:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
            continue
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        length = int(np.hypot(x2 - x1, y2 - y1) / 2)
        ang = int(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        cv2.ellipse(canvas, (mx, my), (length, 4), ang, 0, 360, color, -1)
    if not thin:
        for k, color in enumerate(OP_COLORS):
            cv2.circle(canvas, tuple(pts[k]), 4, color, -1)
    return cv2.addWeighted(frame, 0.4, canvas, 0.6, 0)


def dp_overlay(frame, partmap, nparts):
    """Official DensePose overlay: val = part * 255/nparts, PARULA, alpha 0.7."""
    H, W = frame.shape[:2]
    pm = cv2.resize(partmap.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    mask = pm > 0
    vis = cv2.applyColorMap(np.clip(pm * (255.0 / nparts), 0, 255).astype(np.uint8), DP_CMAP)
    out = frame.copy()
    out[mask] = (frame[mask] * (1 - DP_ALPHA) + vis[mask] * DP_ALPHA).astype(np.uint8)
    return out


def to_camera(Jc, R, pelvis, h):
    """Invert canonicalization: J_cam = R^T (J_canon * h) + pelvis."""
    return np.einsum("ji,kj->ki", R, Jc * h) + pelvis


def draw_overlay(frame, J_cam, K, color, scale):
    """Perspective-project camera-frame joints onto the frame (CoMotion K)."""
    z = np.clip(J_cam[:, 2:3], 1e-3, None)
    uv = (J_cam[:, :2] / z) * K[0] + K[1]
    pts = [(int(u * scale), int(v * scale)) for u, v in uv]
    for c, p in BONES:
        cv2.line(frame, pts[c], pts[p], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, color, -1)


def main(a):
    p = lambda n: os.path.join(a.data, n)
    cho, _ = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    video = p(f"{a.holdout}_video.avi")
    if not os.path.exists(video):
        video = p(f"{a.holdout}.mp4")

    # skeleton: pose from WiFi, per-frame placement (R_can, pelvis) from the
    # teacher, both projected onto the video with CoMotion's default intrinsics
    Y = np.load(p(f"{a.holdout}_Y.npz"), allow_pickle=True)
    k = valid_mask(cho, Y["label_ts"])
    Jt = Y["J_canon"][k].astype(np.float32)
    R, pelvis, h = Y["R_can"][k], Y["pelvis"][k], float(Y["height"])
    fidx = Y["frame_idx"][k]
    Jp = np.load(p("skeleton_holdout_pred.npy"))
    rep = json.load(open(p("skeleton_report.json")))
    n = min(len(Jp), len(Jt))
    cap = cv2.VideoCapture(video)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    K = (2 * max(W, H), np.array([W / 2, H / 2]))  # fx=fy=2*max(H,W), center pp
    LW, LH = 1280, 720
    scale = LW / W
    out = cv2.VideoWriter(p("skeleton_holdout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          30.0, (LW, LH))
    for i in range(n):
        cap.set(1, int(fidx[i]))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (LW, LH))
        draw_overlay(frame, to_camera(Jt[i], R[i], pelvis[i], h), K, (160, 160, 160), scale)
        draw_overlay(frame, to_camera(Jp[i], R[i], pelvis[i], h), K, (0, 220, 120), scale)
        cv2.putText(frame, f"wifi (green) vs teacher (gray)  MPJPE {rep['mpjpe']:.0f}mm  "
                    f"wrist-z r={rep['wrist_z_r']:.2f}", (10, LH - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 1, cv2.LINE_AA)
        out.write(frame)
    cap.release(); out.release()
    print(f"skeleton_holdout.mp4 ({n} frames)")

    # openpose: predictions are at video frame times where the teacher detected
    if os.path.exists(p("openpose_holdout_pred.npy")):
        kp = np.load(p(f"{a.holdout}_kp.npz"))["kp"]
        fts = np.load(p(f"{a.holdout}_frame_ts.npy")).astype(np.float64)
        m = min(len(fts), len(kp))
        k = valid_mask(cho, fts[:m]) & (kp[:m, :, 2].max(1) > 0)
        kf = np.where(k)[0]
        Kt = kp[:m][k][:, :, :2]
        root = 0.5 * (Kt[:, 11] + Kt[:, 12])
        torso = np.linalg.norm(0.5 * (Kt[:, 5] + Kt[:, 6]) - root, axis=1) + 1e-6
        Kp = np.load(p("openpose_holdout_pred.npy"))  # root-relative, torso units
        Kp = Kp * torso[:len(Kp), None, None] + root[:len(Kp), None]
        orep = json.load(open(p("openpose_report.json")))
        cap = cv2.VideoCapture(video)
        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        LW, LH = 1280, 720
        scale = LW / W
        out = cv2.VideoWriter(p("openpose_holdout.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (LW, LH))
        for j in range(min(len(kf), len(Kp))):
            cap.set(1, int(kf[j]))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (LW, LH))
            frame = draw_openpose(frame, to_coco18(Kt[j]), scale, thin=True)
            frame = draw_openpose(frame, to_coco18(Kp[j]), scale)
            cv2.putText(frame, f"wifi (openpose colors) vs teacher (white)  "
                        f"PCK@0.2 {orep['pck20']:.2f}  wrist-y r={orep['wrist_y_r']:.2f}",
                        (10, LH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out.write(frame)
        cap.release(); out.release()
        print("openpose_holdout.mp4")

    # densepose: official detectron2 visualizer overlay, GT (24-part) vs ours
    dp = np.load(p(f"{a.holdout}_dp.npz"))["dp"]
    fts = np.load(p(f"{a.holdout}_frame_ts.npy")).astype(np.float64)
    m = min(len(fts), len(dp))
    k = valid_mask(cho, fts[:m], margin=0.35)
    kf = np.where(k)[0]
    TD = dp[:m][k]
    pred = np.load(p("densepose_holdout_pred.npy"))
    drep = json.load(open(p("densepose_report.json")))
    cap = cv2.VideoCapture(video)
    PW, PH = 560, 315
    out = cv2.VideoWriter(p("densepose_holdout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          30.0, (PW * 3, PH))
    for j in range(min(len(kf), len(pred))):
        cap.set(1, int(kf[j]))
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.resize(frame, (PW, PH))
        gt = dp_overlay(rgb, TD[j], 24)
        pm = dp_overlay(rgb, pred[j], 4)
        for im, lb in [(rgb, "input"), (gt, "DensePose GT (detectron2)"),
                       (pm, f"WiFi (ours)  fg-IoU {drep['fg_iou']:.2f}")]:
            cv2.putText(im, lb, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        out.write(np.hstack([rgb, gt, pm]))
    cap.release(); out.release()
    print("densepose_holdout.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="demo")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

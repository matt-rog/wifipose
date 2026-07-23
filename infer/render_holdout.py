#!/usr/bin/env python3
"""Render holdout predictions next to the camera and teacher ground truth.

python infer/render_holdout.py --holdout holdout --mac <bssid> --data data
"""
import argparse, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wifipose.csi import load_csi
from wifipose.dfs import dfs_features, valid_mask
from wifipose.metrics import PARENTS, PELV
from wifipose.project import smpl_keypoints_2d

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
    kp[14:] = kp[0]  # SMPL teacher has no eyes/ears; collapse face to the head
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
        cv2.ellipse(canvas, (mx, my), (max(length, 1), 6), ang, 0, 360, color, -1)
    if not thin:
        for k, color in enumerate(OP_COLORS):
            cv2.circle(canvas, tuple(pts[k]), 5, color, -1)
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


def draw_skel(panel, J, color, cx, cy, sc):
    """Root-centered canonical-frame skeleton (x right, z up)."""
    pts = [(int(cx + J[j, 0] * sc), int(cy - J[j, 2] * sc)) for j in range(24)]
    for c, p in BONES:
        cv2.line(panel, pts[c], pts[p], color, 2)
    for x, y in pts:
        cv2.circle(panel, (x, y), 2, color, -1)


def doppler_strip(S, ts, t_now, width, height, span=5.0):
    """Scrolling Doppler waterfall, current frame pinned at the right edge."""
    lo = np.searchsorted(ts, t_now - span)
    hi = np.searchsorted(ts, t_now) + 1
    win = S[max(lo, 0):hi]
    if len(win) < 2:
        win = S[max(hi - 2, 0):hi]
    img = cv2.resize(win.T[::-1], (width, height), interpolation=cv2.INTER_LINEAR)
    img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.line(img, (width - 3, 0), (width - 3, height), (255, 255, 255), 2)
    cv2.putText(img, "now", (width - 46, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "wifi doppler", (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ema(X, a=0.3):
    out = X.copy()
    for i in range(1, len(X)):
        out[i] = a * X[i] + (1 - a) * out[i - 1]
    return out


def main(a):
    p = lambda n: os.path.join(a.data, n)
    cho, _ = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    video = p(f"{a.holdout}_video.avi")
    if not os.path.exists(video):
        video = p(f"{a.holdout}.mp4")

    cho2, aho = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    fts_all = np.load(p(f"{a.holdout}_frame_ts.npy")).astype(np.float64)
    kspec = valid_mask(cho2, fts_all)
    spec_ts = fts_all[kspec]
    spec = dfs_features(cho2, aho, spec_ts)[:, 2:28]     # skip near-DC bins
    spec = np.log1p(spec)
    lo_p, hi_p = np.percentile(spec, 10), np.percentile(spec, 99.5)
    spec = np.clip((spec - lo_p) / (hi_p - lo_p + 1e-9), 0, 1) ** 0.7
    STRIP = 90

    # skeleton: root-relative comparison, no teacher information in the
    # prediction rendering (both skeletons centered at their own root)
    Y = np.load(p(f"{a.holdout}_Y.npz"), allow_pickle=True)
    k = valid_mask(cho, Y["label_ts"])
    Jt = Y["J_canon"][k].astype(np.float32)
    fidx = Y["frame_idx"][k]
    Jp = ema(np.load(p("skeleton_holdout_pred.npy")))
    rep = json.load(open(p("skeleton_report.json")))
    n = min(len(Jp), len(Jt))
    cap = cv2.VideoCapture(video)
    LW, LH, PW = 640, 360, 400
    out = cv2.VideoWriter(p("skeleton_holdout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          30.0, (LW + PW, LH + STRIP))
    cx, cy, sc = PW // 2, int(LH * 0.62), 150
    ts_lab = Y["label_ts"].astype(np.float64)[k]
    for i in range(n):
        cap.set(1, int(fidx[i]))
        ok, frame = cap.read()
        if not ok:
            break
        left = cv2.resize(frame, (LW, LH))
        cv2.putText(left, "camera", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.full((LH, PW, 3), 20, np.uint8)
        draw_skel(panel, Jt[i] - Jt[i, PELV], (150, 150, 150), cx, cy, sc)
        draw_skel(panel, Jp[i] - Jp[i, PELV], (0, 220, 120), cx, cy, sc)
        cv2.putText(panel, "green: WiFi prediction (SMPL-24, root-relative)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, "gray: CoMotion teacher (camera)", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        top = np.hstack([left, panel])
        strip = doppler_strip(spec, spec_ts, ts_lab[i], top.shape[1], STRIP)
        out.write(np.vstack([top, strip]))
    cap.release(); out.release()
    print(f"skeleton_holdout.mp4 ({n} frames)")

    # openpose: teacher keypoints projected from the amodal SMPL teacher
    if os.path.exists(p("openpose_holdout_pred.npy")):
        kp = smpl_keypoints_2d(Y["J_canon"].astype(np.float32), Y["R_can"],
                               Y["pelvis"], float(Y["height"]))
        k = valid_mask(cho, Y["label_ts"].astype(np.float64))
        kf = Y["frame_idx"][k]
        Kt = kp[k][:, :, :2]
        root = 0.5 * (Kt[:, 11] + Kt[:, 12])
        torso = np.linalg.norm(0.5 * (Kt[:, 5] + Kt[:, 6]) - root, axis=1) + 1e-6
        Kt_rel = (Kt - root[:, None]) / torso[:, None, None]
        Kp = ema(np.load(p("openpose_holdout_pred.npy")))  # root-relative, torso units
        orep = json.load(open(p("openpose_report.json")))
        cap = cv2.VideoCapture(video)
        LW, LH, PW = 640, 360, 400
        cx, cy, sc = PW // 2, int(LH * 0.42), 95
        out = cv2.VideoWriter(p("openpose_holdout.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (LW + PW, LH + STRIP))
        ts_lab2 = Y["label_ts"].astype(np.float64)[k]
        for j in range(min(len(kf), len(Kp))):
            cap.set(1, int(kf[j]))
            ok, frame = cap.read()
            if not ok:
                break
            left = cv2.resize(frame, (LW, LH))
            cv2.putText(left, "camera", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panel = np.full((LH, PW, 3), 20, np.uint8)
            panel = draw_openpose(panel, to_coco18(Kt_rel[j]) * sc + [cx, cy], 1.0,
                                  thin=True)
            panel = draw_openpose(panel, to_coco18(Kp[j]) * sc + [cx, cy], 1.0)
            cv2.putText(panel, "color: WiFi prediction (COCO keypoints)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "white: SMPL-projected teacher (camera)", (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            top = np.hstack([left, panel])
            strip = doppler_strip(spec, spec_ts, ts_lab2[j], top.shape[1], STRIP)
            out.write(np.vstack([top, strip]))
        cap.release(); out.release()
        print("openpose_holdout.mp4")

    # densepose: official detectron2 visualizer overlay, GT (24-part) vs ours
    dp = np.load(p(f"{a.holdout}_dp.npz"))["dp"]
    fts = np.load(p(f"{a.holdout}_frame_ts.npy")).astype(np.float64)
    m = min(len(fts), len(dp))
    k = valid_mask(cho, fts[:m], margin=0.35)
    kf = np.where(k)[0]
    TD = dp[:m][k]
    drep = json.load(open(p("densepose_report.json")))
    # temporal EMA on class probabilities + bilinear upsampling before argmax:
    # smooth boundaries and no frame flicker (display only; metrics use raw argmax)
    probs = np.load(p("densepose_holdout_probs.npy")).astype(np.float32)
    for j in range(1, len(probs)):
        probs[j] = 0.35 * probs[j] + 0.65 * probs[j - 1]
    COARSE_VAL = np.array([0, 2, 24, 19, 8], np.uint8)  # coarse -> representative
    # fine part id, so both panels share the 24-part colormap scale
    cap = cv2.VideoCapture(video)
    PW, PH = 560, 315
    out = cv2.VideoWriter(p("densepose_holdout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          30.0, (PW * 3, PH + STRIP))
    ts_frames = fts[:m][k]
    for j in range(min(len(kf), len(probs))):
        cap.set(1, int(kf[j]))
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.resize(frame, (PW, PH))
        gt = dp_overlay(rgb, TD[j], 24)
        up = np.stack([cv2.GaussianBlur(
            cv2.resize(probs[j, c], (PW, PH), interpolation=cv2.INTER_LINEAR),
            (31, 31), 0) for c in range(5)])
        pm = dp_overlay(rgb, COARSE_VAL[up.argmax(0)], 24)
        for im, lb in [(rgb, "camera"), (gt, "detectron2 DensePose teacher (camera)"),
                       (pm, "WiFi prediction (coarse 5-class)")]:
            cv2.putText(im, lb, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        top = np.hstack([rgb, gt, pm])
        strip = doppler_strip(spec, spec_ts, ts_frames[j], top.shape[1], STRIP)
        out.write(np.vstack([top, strip]))
    cap.release(); out.release()
    print("densepose_holdout.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="holdout")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

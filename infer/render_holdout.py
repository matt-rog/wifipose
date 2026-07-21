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
PALETTE = np.array([[20, 20, 20], [200, 80, 80], [80, 200, 80],
                    [80, 80, 240], [220, 200, 60]], np.uint8)
CMAP = np.zeros(25, np.int64)
CMAP[[1, 2]] = 1
CMAP[[23, 24]] = 2
CMAP[[3, 4, 15, 16, 17, 18, 19, 20, 21, 22]] = 3
CMAP[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] = 4


def draw(panel, J, color, cx, cy, sc):
    pts = [(int(cx + J[j, 0] * sc), int(cy - J[j, 2] * sc)) for j in range(24)]
    for c, p in BONES:
        cv2.line(panel, pts[c], pts[p], color, 2)
    for x, y in pts:
        cv2.circle(panel, (x, y), 2, color, -1)


def main(a):
    p = lambda n: os.path.join(a.data, n)
    cho, _ = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    video = p(f"{a.holdout}_video.avi")
    if not os.path.exists(video):
        video = p(f"{a.holdout}.mp4")

    # skeleton: predictions are at teacher label times
    Y = np.load(p(f"{a.holdout}_Y.npz"), allow_pickle=True)
    k = valid_mask(cho, Y["label_ts"])
    Jt = Y["J_canon"][k].astype(np.float32)
    fidx = Y["frame_idx"][k]
    Jp = np.load(p("skeleton_holdout_pred.npy"))
    rep = json.load(open(p("skeleton_report.json")))
    n = min(len(Jp), len(Jt))
    cap = cv2.VideoCapture(video)
    LW, LH, PW = 640, 360, 400
    out = cv2.VideoWriter(p("skeleton_holdout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          30.0, (LW + PW, LH))
    cx, cy, sc = PW // 2, int(LH * 0.62), 150
    for i in range(n):
        cap.set(1, int(fidx[i]))
        ok, frame = cap.read()
        if not ok:
            break
        left = cv2.resize(frame, (LW, LH))
        panel = np.full((LH, PW, 3), 20, np.uint8)
        draw(panel, Jt[i] - Jt[i, PELV], (150, 150, 150), cx, cy, sc)
        draw(panel, Jp[i] - Jp[i, PELV], (0, 220, 120), cx, cy, sc)
        cv2.putText(panel, f"MPJPE {rep['mpjpe']:.0f}mm  wrist-z r={rep['wrist_z_r']:.2f}",
                    (8, LH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 220, 120), 1)
        out.write(np.hstack([left, panel]))
    cap.release(); out.release()
    print(f"skeleton_holdout.mp4 ({n} frames)")

    # densepose: predictions are at video frame times
    dp = np.load(p(f"{a.holdout}_dp.npz"))["dp"]
    fts = np.load(p(f"{a.holdout}_frame_ts.npy")).astype(np.float64)
    m = min(len(fts), len(dp))
    k = valid_mask(cho, fts[:m], margin=0.35)
    kf = np.where(k)[0]
    TD = CMAP[dp[:m][k][:, ::10, ::10]]
    pred = np.load(p("densepose_holdout_pred.npy"))
    drep = json.load(open(p("densepose_report.json")))
    cap = cv2.VideoCapture(video)
    out = cv2.VideoWriter(p("densepose_holdout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          30.0, (640 + 640, 360))
    for j in range(min(len(kf), len(pred))):
        cap.set(1, int(kf[j]))
        ok, frame = cap.read()
        if not ok:
            break
        left = cv2.resize(frame, (640, 360))
        gt = cv2.resize(PALETTE[TD[j]], (320, 360), interpolation=cv2.INTER_NEAREST)
        pm = cv2.resize(PALETTE[pred[j]], (320, 360), interpolation=cv2.INTER_NEAREST)
        cv2.putText(gt, "GT (detectron2)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        cv2.putText(pm, f"pred fg-IoU {drep['fg_iou']:.2f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out.write(np.hstack([left, gt, pm]))
    cap.release(); out.release()
    print("densepose_holdout.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="demo")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

#!/usr/bin/env python3
"""Measure the constant video<->CSI time offset from a calibration take
(still 5s, one sharp jumping jack, still 5s, x3). Cross-correlates camera
frame-difference energy against CSI amplitude variance.

python record/sync_offset.py --prefix sync --mac <bssid>
"""
import argparse, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wifipose.csi import load_csi

GRID, MAXLAG = 0.02, 1.0


def main(a):
    cts, amp = load_csi(f"{a.prefix}_csi.npz", a.mac)
    vts = np.load(f"{a.prefix}_frame_ts.npy")
    cap = cv2.VideoCapture(f"{a.prefix}_video.avi")
    vmot, prev = [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        vmot.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    n = min(len(vmot), len(vts))
    vmot, vts = np.array(vmot[:n]), vts[:n]

    grid = np.arange(max(cts[0], vts[0]) + 0.3, min(cts[-1], vts[-1]) - 0.3, GRID)
    cmot = np.empty(len(grid))
    for i, t in enumerate(grid):
        w = amp[np.searchsorted(cts, t - 0.1):np.searchsorted(cts, t + 0.1)]
        cmot[i] = w.std(0).mean() if len(w) > 4 else 0.0
    vmot_g = np.interp(grid, vts, vmot)

    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    zc, zv = z(cmot), z(vmot_g)
    lags = np.arange(-int(MAXLAG / GRID), int(MAXLAG / GRID) + 1)
    xc = np.array([np.corrcoef(zc[max(0, l):len(zc) + min(0, l)],
                               zv[max(0, -l):len(zv) - max(0, l)])[0, 1] for l in lags])
    best = int(np.argmax(xc))
    offset = -lags[best] * GRID
    peak, med = xc[best], float(np.median(xc))
    print(f"peak r={peak:.2f} at video-minus-CSI offset {offset * 1000:+.0f} ms "
          f"(median {med:.2f})")
    if peak < 0.4 or peak - med < 0.15:
        print("WARNING: weak/flat peak, redo the take with sharper onsets")
    else:
        print(f"use --sync-offset {offset:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sync")
    ap.add_argument("--mac", required=True)
    main(ap.parse_args())

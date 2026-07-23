#!/usr/bin/env python3
"""CoMotion track .pt -> canonicalized SMPL-24 joint targets.

Per frame: dominant track -> SMPL forward (apple/ml-comotion SMPLKinematics)
-> root-relative -> yaw-canonical -> height-normalized. Emits per-frame quality
weights, gap interpolation flags, per-joint loss weights, and the inverse
transform (R_can, pelvis, h) for rendering back to camera space.

python teacher/comotion_targets.py --pt A.pt --frame-ts A_frame_ts.npy \
    --sync-offset 0.100 --out A_Y.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("COMOTION_SRC", "ml-comotion/src")))

# ---- SMPL-24 kinematic indices (standard SMPL joint order) ----
PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD = 0, 3, 6, 9, 12, 15
L_HIP, R_HIP = 1, 2
L_SHO, R_SHO = 16, 17
L_ELB, R_ELB = 18, 19
L_WRI, R_WRI = 20, 21
LEGS = [4, 5, 7, 8, 10, 11]           # knees, ankles, feet
HANDS = [22, 23]

MAX_GAP = 3                            # interpolate teacher gaps <= this many frames


def lambda_joint():
    """Per-joint loss weight — spend budget on the upper chain (waving demo)."""
    lam = np.ones(24, dtype=np.float32)
    lam[[L_WRI, R_WRI]] = 3.5
    lam[[L_ELB, R_ELB]] = 2.0
    lam[[L_SHO, R_SHO]] = 1.5
    lam[HANDS] = 1.0                   # SMPL hand tips are noisy — don't over-trust
    lam[LEGS] = 0.2                    # standing subject: legs barely move, mostly noise
    return lam


def decode_smpl(pose, betas, trans):
    """CoMotion SMPL params -> [N,24,3] joints in camera frame (reuses SMPLKinematics)."""
    from comotion_demo.utils.smpl_kinematics import SMPLKinematics
    smpl = SMPLKinematics().eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(pose), 512):
            b = slice(s, s + 512)
            J = smpl(torch.from_numpy(betas[b]).float(),
                     torch.from_numpy(pose[b]).float(),
                     torch.from_numpy(trans[b]).float(),
                     output_format="joints")           # [n,24,3]
            out.append(J.numpy())
    return np.concatenate(out).astype(np.float32)


def fill_gaps(frames, J, pose, betas):
    """Linear-interp joint xyz across teacher gaps <= MAX_GAP; flag interpolated
    frames (coverage weight 0.5). Gaps > MAX_GAP left as holes (window dropped
    later by the pairing builder). Returns dense frames + arrays + interp mask."""
    order = np.argsort(frames)
    frames, J = frames[order], J[order]
    pose, betas = pose[order], betas[order]
    lo, hi = int(frames[0]), int(frames[-1])
    dense = np.arange(lo, hi + 1)
    have = {int(f): i for i, f in enumerate(frames)}
    Jd = np.zeros((len(dense), 24, 3), np.float32)
    Pd = np.zeros((len(dense), pose.shape[1]), np.float32)
    Bd = np.zeros((len(dense), betas.shape[1]), np.float32)
    interp = np.zeros(len(dense), bool)
    valid = np.ones(len(dense), bool)
    # locate real frames, then interp holes
    real_pos = np.array([k for k, f in enumerate(dense) if int(f) in have])
    real_f = dense[real_pos]
    for k, f in enumerate(dense):
        if int(f) in have:
            i = have[int(f)]
            Jd[k], Pd[k], Bd[k] = J[i], pose[i], betas[i]
        else:
            # nearest real frames on each side
            left = real_f[real_f < f].max() if (real_f < f).any() else None
            right = real_f[real_f > f].min() if (real_f > f).any() else None
            gap = (right - left - 1) if (left is not None and right is not None) else 999
            if left is None or right is None or gap > MAX_GAP:
                valid[k] = False
                continue
            a = (f - left) / (right - left)
            Jd[k] = (1 - a) * J[have[int(left)]] + a * J[have[int(right)]]
            Pd[k] = (1 - a) * pose[have[int(left)]] + a * pose[have[int(right)]]
            Bd[k] = betas[have[int(left)]]           # betas ~const; hold
            interp[k] = True
    return dense[valid], Jd[valid], Pd[valid], Bd[valid], interp[valid]


def canonicalize(J):
    """root-relative + yaw-canonical + height-normalized. Returns J_canon[N,24,3],
    plus per-frame invert constants (R_can[N,3,3], pelvis[N,3]) and scalar h.

    Gravity/up is ESTIMATED from the median spine direction (pelvis->neck) across
    frames, NOT hardcoded — CoMotion is a camera frame (y-down), so a hardcoded
    [0,0,1] would flip everyone upside down (research caveat)."""
    pelvis = J[:, PELVIS].copy()                      # [N,3]
    Jr = J - pelvis[:, None]                           # root-relative
    # up = median spine direction over all frames (robust to per-frame wobble)
    spine = J[:, NECK] - J[:, PELVIS]
    up = np.median(spine, axis=0)
    up = up / (np.linalg.norm(up) + 1e-9)
    # per-frame right vector from hips (R - L), projected orthogonal to up
    hip = Jr[:, R_HIP] - Jr[:, L_HIP]                  # [N,3]
    right = hip - (hip @ up)[:, None] * up
    right /= (np.linalg.norm(right, axis=1, keepdims=True) + 1e-9)
    fwd = np.cross(np.broadcast_to(up, right.shape), right)
    fwd /= (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9)
    R_can = np.stack([right, fwd, np.broadcast_to(up, right.shape)], axis=1)  # [N,3,3] world->canon
    Jc = np.einsum("nij,nkj->nki", R_can, Jr).astype(np.float32)              # rotate each joint
    # height normalization: median torso length (pelvis->neck)
    h = float(np.median(np.linalg.norm(J[:, NECK] - J[:, PELVIS], axis=1)))
    Jc /= (h + 1e-9)
    return Jc, R_can.astype(np.float32), pelvis.astype(np.float32), h


def proxy_weights(J, interp):
    """No native teacher confidence -> proxy from temporal jerk + coverage.
    Soft-weight (don't hard-filter). Interpolated frames capped at 0.5."""
    # second difference (jerk) magnitude, mean over joints
    d2 = np.zeros(len(J), np.float32)
    if len(J) > 2:
        acc = J[2:] - 2 * J[1:-1] + J[:-2]
        jerk = np.linalg.norm(acc, axis=2).mean(1)
        d2[1:-1] = jerk
    s = np.median(d2[d2 > 0]) + 1e-6
    w = np.exp(-d2 / (3 * s)).astype(np.float32)       # teleporting joints -> down-weight
    w[interp] = np.minimum(w[interp], 0.5)
    return w


def main(a):
    d = torch.load(a.pt, map_location="cpu", weights_only=True)
    ids = d["id"].numpy().astype(int)
    fr = d["frame_idx"].numpy().astype(int)
    pose = d["pose"].numpy().astype(np.float32)
    trans = d["trans"].numpy().astype(np.float32)
    betas = d["betas"].numpy().astype(np.float32)

    # dominant track = most-frequent id
    dom = np.bincount(ids - ids.min()).argmax() + ids.min()
    m = ids == dom
    print(f"tracks {np.unique(ids)} -> dominant id {dom}: "
          f"{m.sum()}/{len(ids)} frames ({100*m.mean():.0f}%)")
    fr, pose, trans, betas = fr[m], pose[m], trans[m], betas[m]

    J = decode_smpl(pose, betas, trans)                # [N,24,3] camera frame
    frames, J, pose, betas, interp = fill_gaps(fr, J, pose, betas)
    print(f"after gap-fill (<= {MAX_GAP}): {len(frames)} frames, "
          f"{int(interp.sum())} interpolated")

    Jc, R_can, pelvis, h = canonicalize(J)
    w = proxy_weights(J, interp)
    presence = np.zeros(len(frames), np.float32) if a.empty else np.ones(len(frames), np.float32)

    # frame_ts (monotonic, sync-corrected) if provided — enables timestamp pairing
    label_ts = None
    if a.frame_ts:
        fts = np.load(a.frame_ts)
        label_ts = fts[frames] - a.sync_offset

    out = dict(frame_idx=frames.astype(np.int64), J_canon=Jc, presence=presence,
               w_frame=w, interp=interp, lambda_joint=lambda_joint(),
               R_can=R_can, pelvis=pelvis, height=np.float32(h),
               track_id=np.int64(dom), empty=np.bool_(a.empty))
    if label_ts is not None:
        out["label_ts"] = label_ts.astype(np.float64)
    np.savez(a.out, **out)

    # ---- report ----
    d015 = np.linalg.norm(J[:, HEAD] - J[:, PELVIS], axis=1)
    wrist_mot = np.linalg.norm(np.diff(Jc[:, R_WRI], axis=0), axis=1)
    print(f"saved {a.out}")
    print(f"  torso height h = {h:.3f} m")
    print(f"  pelvis->head raw: {d015.mean():.3f}±{d015.std():.3f} m")
    print(f"  canon R-wrist per-frame motion (norm units): "
          f"mean {wrist_mot.mean():.3f} max {wrist_mot.max():.3f}  "
          f"(>0 confirms wave survives canonicalization)")
    print(f"  w_frame: mean {w.mean():.3f} min {w.min():.3f} "
          f"(<0.5 frames: {int((w < 0.5).sum())})")
    print(f"  presence = {'0 (EMPTY)' if a.empty else '1 (occupied)'}")
    if label_ts is not None:
        print(f"  label_ts span {label_ts[-1]-label_ts[0]:.1f}s "
              f"[{label_ts[0]:.3f} .. {label_ts[-1]:.3f}]")
    # sanity: canonical frame should NOT be upside down — head above pelvis (+z)
    head_up = Jc[:, HEAD, 2].mean()
    print(f"  canon head z (should be POSITIVE = up): {head_up:+.3f}"
          f"  {'OK' if head_up > 0 else '*** FLIPPED — check gravity axis ***'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="CoMotion track .pt")
    ap.add_argument("--frame-ts", dest="frame_ts", default=None,
                    help="frame_ts .npy (monotonic, same clock as CSI) for pairing")
    ap.add_argument("--sync-offset", dest="sync_offset", type=float, default=0.0,
                    help="video-minus-CSI seconds (from sync_offset.py)")
    ap.add_argument("--empty", action="store_true",
                    help="empty-room recording: presence=0, no pose target")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())

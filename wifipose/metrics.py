"""Pose/segmentation metrics: MPJPE, PA-MPJPE, torso-PCK (MM-Fi conventions),
plus constant-baseline, wrist correlation, and variance-ratio honesty checks."""
import numpy as np

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19, 20, 21]
PELV, NECK, L_WRI, R_WRI = 0, 12, 20, 21
HEIGHT_REF = 0.546  # meters per canonical unit


def procrustes_align(P, T):
    out = np.empty_like(P)
    for i in range(len(P)):
        p0, t0 = P[i] - P[i].mean(0), T[i] - T[i].mean(0)
        U, S, Vt = np.linalg.svd(t0.T @ p0)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, 1, d]) @ Vt
        s = (S[:2].sum() + d * S[2]) / (np.square(p0).sum() + 1e-12)
        out[i] = s * (p0 @ R.T) + T[i].mean(0)
    return out


def pose_report(Jp, Jt):
    n = min(len(Jp), len(Jt))
    P = Jp[:n] - Jp[:n, PELV:PELV + 1]
    T = Jt[:n] - Jt[:n, PELV:PELV + 1]
    mm = HEIGHT_REF * 1000
    torso = np.linalg.norm(T[:, NECK] - T[:, PELV], axis=1) + 1e-6
    dd = np.linalg.norm(P - T, axis=2)

    def wc(j, ax):
        return np.corrcoef(P[:, j, ax], T[:, j, ax])[0, 1]

    return dict(
        mpjpe=float(dd.mean() * mm),
        pa_mpjpe=float(np.linalg.norm(procrustes_align(P, T) - T, axis=2).mean() * mm),
        pck20=float((dd < 0.2 * torso[:, None]).mean()),
        wrist_r=float(np.nanmean([wc(j, a) for j in (L_WRI, R_WRI) for a in range(3)])),
        wrist_z_r=float(0.5 * (wc(L_WRI, 2) + wc(R_WRI, 2))),
        pred_var_ratio=float(max(P[:, :, a].std(0).mean() / (T[:, :, a].std(0).mean() + 1e-9)
                                 for a in range(3))),
    )


def constant_baseline_mpjpe(Jtrain, Jt):
    n = len(Jt)
    C = np.tile(Jtrain.mean(0) - Jtrain.mean(0)[PELV], (n, 1, 1))
    T = Jt - Jt[:, PELV:PELV + 1]
    return float(np.linalg.norm(C - T, axis=2).mean() * HEIGHT_REF * 1000)


def seg_report(pred, targ, nclass=5):
    """fg-IoU, part-mIoU, and matched vs time-shuffled per-frame fg-IoU
    (their gap isolates genuine per-frame tracking from a static blob)."""
    fg = float(((pred > 0) & (targ > 0)).sum() / (((pred > 0) | (targ > 0)).sum() + 1e-9))
    per = [((pred == c) & (targ == c)).sum() / (((pred == c) | (targ == c)).sum() + 1e-9)
           for c in range(1, nclass) if (targ == c).sum() > 0]
    perm = np.random.default_rng(0).permutation(len(targ))

    def f(a, b):
        return ((a > 0) & (b > 0)).sum() / (((a > 0) | (b > 0)).sum() + 1e-9)

    matched = float(np.mean([f(pred[i], targ[i]) for i in range(len(targ))]))
    shuffled = float(np.mean([f(pred[i], targ[perm[i]]) for i in range(len(targ))]))
    return dict(fg_iou=fg, part_miou=float(np.mean(per)) if per else 0.0,
                frame_fg_iou=matched, shuffled_fg_iou=shuffled,
                tracking_gap=matched - shuffled)

#!/usr/bin/env python3
"""Train DFS -> COCO-17 2D keypoints, root-relative and torso-normalized
(absolute position is not observable from CSI), evaluated on a separate-
recording holdout. Keypoints are projected from the amodal SMPL teacher, which
infers off-frame body parts; per-frame keypoint detectors hallucinate them.
PCK follows the Person-in-WiFi convention.

python train/train_openpose.py --train train --holdout holdout --mac <bssid> --data data
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wifipose.csi import load_csi
from wifipose.dfs import dfs_features, jittered_features, valid_mask
from wifipose.project import smpl_keypoints_2d

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS, EPOCHS, PATIENCE, BS = 5, 300, 40, 256
L_SHO, R_SHO, L_WRI, R_WRI, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12
LAM17 = np.array([.5, .5, .5, .5, .5, 1.5, 1.5, 2, 2, 3.5, 3.5,
                  1, 1, .2, .2, .2, .2], np.float32)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.4),
                               nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.4))
        self.p = nn.Linear(128, 34)

    def forward(self, x):
        return self.p(self.t(x))


def augment(xb, jit, idx):
    B, d = len(idx), xb.shape[1]
    src = torch.randint(jit.shape[0] + 1, (B,), device=DEV)
    for j in range(jit.shape[0]):
        m = src == (j + 1)
        if m.any():
            xb[m] = jit[j][idx[m]]
    xb = xb + 0.05 * torch.randn_like(xb)
    xb = xb * (1.0 + 0.15 * torch.randn(B, 1, device=DEV))
    xb = xb + 0.10 * torch.randn_like(xb) * torch.rand(B, 1, device=DEV)
    for _ in range(2):
        L0 = torch.randint(0, d, (B,), device=DEV)
        Lw = torch.randint(4, int(0.1 * d), (B,), device=DEV)
        ar = torch.arange(d, device=DEV)[None, :]
        xb = xb * ~((ar >= L0[:, None]) & (ar < (L0 + Lw)[:, None]))
    return xb


def pck_report(Kp, Kt, S):
    """PCK@a normalized by torso diameter (mid-shoulder to mid-hip), scored
    keypoints only, plus wrist correlation and variance honesty checks."""
    n = min(len(Kp), len(Kt))
    P, T, vis = Kp[:n], Kt[:n], S[:n] > 0
    torso = np.linalg.norm(0.5 * (T[:, L_SHO] + T[:, R_SHO])
                           - 0.5 * (T[:, L_HIP] + T[:, R_HIP]), axis=1) + 1e-6
    d = np.linalg.norm(P - T, axis=2)

    def wc(j, ax):
        m = vis[:, j]
        return np.corrcoef(P[m, j, ax], T[m, j, ax])[0, 1] if m.sum() > 10 else np.nan

    return dict(
        pck20=float((d[vis] < 0.2 * np.repeat(torso[:, None], 17, 1)[vis]).mean()),
        pck50=float((d[vis] < 0.5 * np.repeat(torso[:, None], 17, 1)[vis]).mean()),
        mean_err_torso=float((d[vis] / np.repeat(torso[:, None], 17, 1)[vis]).mean()),
        wrist_y_r=float(np.nanmean([wc(L_WRI, 1), wc(R_WRI, 1)])),
        wrist_r=float(np.nanmean([wc(j, a) for j in (L_WRI, R_WRI) for a in (0, 1)])),
        pred_var_ratio=float(P[:, :, 1].std(0).mean() / (T[:, :, 1].std(0).mean() + 1e-9)),
    )


def main(a):
    p = lambda n: os.path.join(a.data, n)
    ctr, atr = load_csi(p(f"{a.train}_csi.npz"), a.mac)
    cho, aho = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    Ytr_npz = np.load(p(f"{a.train}_Y.npz"), allow_pickle=True)
    Yho_npz = np.load(p(f"{a.holdout}_Y.npz"), allow_pickle=True)
    ttr_all = Ytr_npz["label_ts"].astype(np.float64)
    tho_all = Yho_npz["label_ts"].astype(np.float64)
    kptr = smpl_keypoints_2d(Ytr_npz["J_canon"].astype(np.float32),
                             Ytr_npz["R_can"], Ytr_npz["pelvis"], float(Ytr_npz["height"]))
    kpho = smpl_keypoints_2d(Yho_npz["J_canon"].astype(np.float32),
                             Yho_npz["R_can"], Yho_npz["pelvis"], float(Yho_npz["height"]))

    ktr = valid_mask(ctr, ttr_all, margin=0.35)
    kho = valid_mask(cho, tho_all)
    XA = dfs_features(ctr, atr, ttr_all[ktr])
    XJ = jittered_features(ctr, atr, ttr_all[ktr])
    XD = dfs_features(cho, aho, tho_all[kho])
    def rel(kp):
        xy, s = kp[:, :, :2].astype(np.float32), kp[:, :, 2]
        root = 0.5 * (xy[:, L_HIP] + xy[:, R_HIP])
        torso = np.linalg.norm(0.5 * (xy[:, L_SHO] + xy[:, R_SHO]) - root,
                               axis=1) + 1e-6
        return (xy - root[:, None]) / torso[:, None, None], s

    Ktr, Sr = rel(kptr[ktr])
    Kho, Sh = rel(kpho[kho])
    print(f"train {len(XA)} frames, holdout {len(XD)} frames", flush=True)

    N, ntr = len(XA), int(0.9 * len(XA))
    fm, fs = XA[:ntr].mean(0), XA[:ntr].std(0) + 1e-8
    Yf = Ktr.reshape(N, 34)
    ym, ys = Yf[:ntr].mean(0), Yf[:ntr].std(0) + 1e-8
    tt = lambda x: torch.tensor(x, device=DEV)
    Xtr, XJtr = tt((XA[:ntr] - fm) / fs), tt((XJ[:, :ntr] - fm) / fs)
    Xv = tt((XA[ntr:] - fm) / fs)
    Ytr, Yv = tt(((Yf - ym) / ys)[:ntr]), tt(((Yf - ym) / ys)[ntr:])
    XDt = tt((XD - fm) / fs)
    w = np.repeat(LAM17, 2)[None] * np.repeat(Sr[:ntr], 2, axis=1)
    Wtr = tt((w / w.mean()).astype(np.float32))
    hub = nn.SmoothL1Loss(beta=0.1, reduction="none")

    preds = []
    for sd in range(SEEDS):
        torch.manual_seed(sd)
        net = MLP(XA.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=3e-2)
        best, bad, state = 1e9, 0, None
        for _ in range(EPOCHS):
            net.train()
            for _ in range(0, ntr, BS):
                idx = torch.randperm(ntr, device=DEV)[:BS]
                L = (hub(net(augment(Xtr[idx].clone(), XJtr, idx)), Ytr[idx])
                     * Wtr[idx]).mean()
                opt.zero_grad(); L.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                ev = float(((net(Xv) - Yv) ** 2).mean())
            if ev < best:
                best, bad, state = ev, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            preds.append((net(XDt).cpu().numpy() * ys + ym).reshape(-1, 17, 2))
        torch.save(dict(state_dict=net.state_dict(), fm=fm, fs=fs, ym=ym, ys=ys),
                   p(f"openpose_seed{sd}.pt"))
    Kp = np.mean(preds, 0)
    rep = pck_report(Kp, Kho, Sh)
    const = np.tile(Ktr[:ntr].mean(0), (len(Kho), 1, 1))
    rep["const_pck20"] = pck_report(const, Kho, Sh)["pck20"]
    np.save(p("openpose_holdout_pred.npy"), Kp)
    json.dump(rep, open(p("openpose_report.json"), "w"), indent=1)
    for k, v in rep.items():
        print(f"{k:16s} {v:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train")
    ap.add_argument("--holdout", default="holdout")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

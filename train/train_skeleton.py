#!/usr/bin/env python3
"""Train DFS -> SMPL-24 skeleton, evaluate on a separate-recording holdout.

python train/train_skeleton.py --train A --holdout demo --mac <bssid> --data data
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wifipose.csi import load_csi
from wifipose.dfs import dfs_features, jittered_features, valid_mask
from wifipose.metrics import PARENTS, pose_report, constant_baseline_mpjpe

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BONES = [(c, PARENTS[c]) for c in range(1, 24)]
SEEDS, EPOCHS, PATIENCE, BS = 5, 300, 40, 256


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.4),
                               nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.4))
        self.p = nn.Linear(128, 72)

    def forward(self, x):
        return self.p(self.t(x))


def augment(xb, jit, idx):
    B, d = len(idx), xb.shape[1]
    src = torch.randint(jit.shape[0] + 1, (B,), device=DEV)
    for j in range(jit.shape[0]):  # time-shift
        m = src == (j + 1)
        if m.any():
            xb[m] = jit[j][idx[m]]
    xb = xb + 0.05 * torch.randn_like(xb)
    xb = xb * (1.0 + 0.15 * torch.randn(B, 1, device=DEV))
    xb = xb + 0.10 * torch.randn_like(xb) * torch.rand(B, 1, device=DEV)
    for _ in range(2):  # frequency masking
        L0 = torch.randint(0, d, (B,), device=DEV)
        Lw = torch.randint(4, int(0.1 * d), (B,), device=DEV)
        ar = torch.arange(d, device=DEV)[None, :]
        xb = xb * ~((ar >= L0[:, None]) & (ar < (L0 + Lw)[:, None]))
    return xb


def train_seed(Xtr, XJtr, Ytr, Xv, Yv, lam, unstd, seed):
    torch.manual_seed(seed)
    ntr = len(Xtr)
    net = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=3e-2)
    hub = nn.SmoothL1Loss(beta=0.1, reduction="none")
    pairs = torch.tensor([[i - 1, i] for i in range(1, ntr)], device=DEV)
    best, bad, state = 1e9, 0, None
    for _ in range(EPOCHS):
        net.train()
        for _ in range(0, ntr, BS):
            idx = torch.randperm(ntr, device=DEV)[:BS]
            pz = net(augment(Xtr[idx].clone(), XJtr, idx))
            L = (hub(pz, Ytr[idx]) * lam).mean()
            pc, yc = unstd(pz).reshape(-1, 24, 3), unstd(Ytr[idx]).reshape(-1, 24, 3)
            lp = torch.stack([(pc[:, c] - pc[:, p]).norm(dim=1) for c, p in BONES], 1)
            ly = torch.stack([(yc[:, c] - yc[:, p]).norm(dim=1) for c, p in BONES], 1)
            L = L + 0.1 * hub(lp, ly).mean()  # bone length
            sel = pairs[torch.randint(len(pairs), (BS,), device=DEV)]
            L = L + 0.5 * hub(unstd(net(Xtr[sel[:, 1]]) - net(Xtr[sel[:, 0]])),
                              unstd(Ytr[sel[:, 1]] - Ytr[sel[:, 0]])).mean()  # velocity
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
    return net


def main(a):
    p = lambda n: os.path.join(a.data, n)
    ctr, atr = load_csi(p(f"{a.train}_csi.npz"), a.mac)
    cho, aho = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    Atr = np.load(p(f"{a.train}_Y.npz"), allow_pickle=True)
    Aho = np.load(p(f"{a.holdout}_Y.npz"), allow_pickle=True)

    ktr = valid_mask(ctr, Atr["label_ts"], margin=0.35)
    kho = valid_mask(cho, Aho["label_ts"])
    ttr, Jtr = Atr["label_ts"][ktr], Atr["J_canon"][ktr].astype(np.float32)
    tho, Jho = Aho["label_ts"][kho], Aho["J_canon"][kho].astype(np.float32)
    XA = dfs_features(ctr, atr, ttr)
    XJ = jittered_features(ctr, atr, ttr)
    XD = dfs_features(cho, aho, tho)
    print(f"train {len(XA)} frames, holdout {len(XD)} frames", flush=True)

    N, ntr = len(XA), int(0.9 * len(XA))  # time-blocked val tail: early stopping only
    fm, fs = XA[:ntr].mean(0), XA[:ntr].std(0) + 1e-8
    Yf = Jtr.reshape(N, 72)
    ym, ys = Yf[:ntr].mean(0), Yf[:ntr].std(0) + 1e-8
    tt = lambda x: torch.tensor(x, device=DEV)
    Xtr, XJtr = tt((XA[:ntr] - fm) / fs), tt((XJ[:, :ntr] - fm) / fs)
    Xv = tt((XA[ntr:] - fm) / fs)
    Ytr, Yv = tt(((Yf - ym) / ys)[:ntr]), tt(((Yf - ym) / ys)[ntr:])
    XDt = tt((XD - fm) / fs)
    lam = tt(np.repeat(Atr["lambda_joint"].astype(np.float32), 3))
    ym_t, ys_t = tt(ym), tt(ys)
    unstd = lambda z: z * ys_t + ym_t

    preds = []
    for sd in range(SEEDS):
        net = train_seed(Xtr, XJtr, Ytr, Xv, Yv, lam, unstd, sd)
        with torch.no_grad():
            preds.append((net(XDt).cpu().numpy() * ys + ym).reshape(-1, 24, 3))
        torch.save(dict(state_dict=net.state_dict(), fm=fm, fs=fs, ym=ym, ys=ys),
                   p(f"skeleton_seed{sd}.pt"))
    Jp = np.mean(preds, 0)
    rep = pose_report(Jp, Jho)
    rep["const_mpjpe"] = constant_baseline_mpjpe(Jtr[:ntr], Jho[:min(len(Jp), len(Jho))])
    np.save(p("skeleton_holdout_pred.npy"), Jp)
    json.dump(rep, open(p("skeleton_report.json"), "w"), indent=1)
    for k, v in rep.items():
        print(f"{k:15s} {v:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="A")
    ap.add_argument("--holdout", default="demo")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

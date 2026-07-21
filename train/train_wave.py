#!/usr/bin/env python3
"""Train DFS -> 1-D arm elevation (mean wrist height), the strongest single
cross-recording signal. No augmentation: it removes the absolute-power cue
this task depends on (ablated 2026-07-21).

python train/train_wave.py --train A --holdout demo --mac <bssid> --data data
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wifipose.csi import load_csi
from wifipose.dfs import dfs_features, valid_mask
from wifipose.metrics import L_WRI, R_WRI

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS, EPOCHS, PATIENCE, BS = 10, 300, 40, 256


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.4),
                               nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.4))
        self.p = nn.Linear(128, 1)

    def forward(self, x):
        return self.p(self.t(x))


def ema(x, a=0.3):
    o = x.copy()
    for i in range(1, len(x)):
        o[i] = a * x[i] + (1 - a) * o[i - 1]
    return o


def main(a):
    p = lambda n: os.path.join(a.data, n)
    ctr, atr = load_csi(p(f"{a.train}_csi.npz"), a.mac)
    cho, aho = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    Atr = np.load(p(f"{a.train}_Y.npz"), allow_pickle=True)
    Aho = np.load(p(f"{a.holdout}_Y.npz"), allow_pickle=True)
    ktr = valid_mask(ctr, Atr["label_ts"])
    kho = valid_mask(cho, Aho["label_ts"])
    XA = dfs_features(ctr, atr, Atr["label_ts"][ktr])
    XD = dfs_features(cho, aho, Aho["label_ts"][kho])

    target = lambda J: 0.5 * (J[:, L_WRI, 2] + J[:, R_WRI, 2])
    Y = target(Atr["J_canon"][ktr].astype(np.float32))[:, None]
    Yho = target(Aho["J_canon"][kho].astype(np.float32))

    ntr = int(0.9 * len(XA))
    fm, fs = XA.mean(0), XA.std(0) + 1e-8
    ym, ys = Y.mean(0), Y.std(0) + 1e-8
    tt = lambda x: torch.tensor(x, device=DEV)
    Xtr, Ytr = tt(((XA - fm) / fs)[:ntr]), tt(((Y - ym) / ys)[:ntr])
    Xv, Yv = tt(((XA - fm) / fs)[ntr:]), tt(((Y - ym) / ys)[ntr:])
    XDt = tt((XD - fm) / fs)
    hub = nn.SmoothL1Loss(beta=0.1)

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
                L = hub(net(Xtr[idx] + 0.05 * torch.randn_like(Xtr[idx])), Ytr[idx])
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
            preds.append(net(XDt).cpu().numpy() * ys + ym)
        torch.save(dict(state_dict=state, fm=fm, fs=fs, ym=ym, ys=ys),
                   p(f"wave_seed{sd}.pt"))

    n = min(len(XD), len(Yho))
    pe = ema(np.mean(preds, 0)[:n, 0])
    r = float(np.corrcoef(pe, Yho[:n])[0, 1])
    np.save(p("wave_holdout_pred.npy"), pe)
    json.dump(dict(pearson_r=r), open(p("wave_report.json"), "w"), indent=1)
    print(f"arm-elevation holdout r = {r:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="A")
    ap.add_argument("--holdout", default="demo")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

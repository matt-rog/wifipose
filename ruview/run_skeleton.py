#!/usr/bin/env python3
"""RuView skeleton baseline: their Net (raw-amplitude MLP, sigmoid output) on
our data and eval. Input is their representation, windowed amplitude
statistics, not Doppler.

python ruview/run_skeleton.py --mac <bssid> --data data
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RuView", "examples", "through-wall"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RuView", "archive", "v1", "src", "models"))
from wifipose.csi import load_csi
from wifipose.metrics import pose_report, constant_baseline_mpjpe
from wiflow_train import Net

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, BS, WIN = 300, 40, 256, 0.25


def amp_features(cts, amp, at_ts):
    """Windowed mean + coefficient-of-variation spectrum (raw amplitude)."""
    out = np.empty((len(at_ts), 2 * amp.shape[1]), np.float32)
    for i, t in enumerate(at_ts):
        lo, hi = np.searchsorted(cts, t - WIN), np.searchsorted(cts, t)
        w = amp[lo:hi] if hi - lo >= 5 else amp[max(0, hi - 5):hi]
        mu = w.mean(0)
        out[i, :amp.shape[1]] = mu / (mu.mean() + 1e-6)
        out[i, amp.shape[1]:] = w.std(0) / (mu + 1e-6)
    return out


def main(a):
    p = lambda n: os.path.join(a.data, n)
    ctr, atr = load_csi(p(f"{a.train}_csi.npz"), a.mac)
    cho, aho = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    Atr = np.load(p(f"{a.train}_Y.npz"), allow_pickle=True)
    Aho = np.load(p(f"{a.holdout}_Y.npz"), allow_pickle=True)
    ktr = (Atr["label_ts"] >= ctr[0] + WIN) & (Atr["label_ts"] <= ctr[-1])
    kho = (Aho["label_ts"] >= cho[0] + WIN) & (Aho["label_ts"] <= cho[-1])
    XA = amp_features(ctr, atr, Atr["label_ts"][ktr])
    XD = amp_features(cho, aho, Aho["label_ts"][kho])
    Jtr = Atr["J_canon"][ktr].astype(np.float32)
    Jho = Aho["J_canon"][kho].astype(np.float32)

    N, ntr = len(XA), int(0.9 * len(XA))
    fm, fs = XA[:ntr].mean(0), XA[:ntr].std(0) + 1e-8
    Y = Jtr.reshape(N, 72)
    ymin, ymax = Y[:ntr].min(0), Y[:ntr].max(0)
    yr = ymax - ymin + 1e-6  # sigmoid output needs [0,1] targets
    tt = lambda x: torch.tensor(x, device=DEV)
    Xtr, Ytr = tt(((XA - fm) / fs)[:ntr]), tt(((Y - ymin) / yr)[:ntr])
    Xv, Yv = tt(((XA - fm) / fs)[ntr:]), tt(((Y - ymin) / yr)[ntr:])
    XDt = tt((XD - fm) / fs)

    torch.manual_seed(0)
    net = Net(XA.shape[1], 72).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, bad, state = 1e9, 0, None
    for _ in range(EPOCHS):
        net.train()
        for s in range(0, ntr, BS):
            idx = torch.randperm(ntr, device=DEV)[:BS]
            L = ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean()
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
        Jp = (net(XDt).cpu().numpy() * yr + ymin).reshape(-1, 24, 3)
    rep = pose_report(Jp, Jho)
    rep["const_mpjpe"] = constant_baseline_mpjpe(Jtr[:ntr], Jho[:min(len(Jp), len(Jho))])
    np.save(p("ruview_skeleton_pred.npy"), Jp)
    json.dump(rep, open(p("ruview_skeleton_report.json"), "w"), indent=1)
    for k, v in rep.items():
        print(f"{k:15s} {v:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train")
    ap.add_argument("--holdout", default="holdout")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

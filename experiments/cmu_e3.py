#!/usr/bin/env python3
"""E3: CMU-style modality translation into a pseudo-image consumed by a frozen
ImageNet-pretrained ResNet18 stem (conv1+bn+layer1), trainable seg head on top.
Tests whether borrowed vision priors help at our scale.
Run from ~/wifipose:  python cmu_e3.py
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
sys.path.insert(0, os.path.expanduser("~/wifipose"))
from wifipose.csi import load_csi
from wifipose.dfs import dfs_features, jittered_features, valid_mask, W_SEC
from wifipose.metrics import seg_report

DEV = torch.device("cuda")
MAC = "0e:d7:a0:26:9d:74"
GH, GW = 36, 64
CMAP = np.zeros(25, np.int64)
CMAP[[1, 2]] = 1; CMAP[[23, 24]] = 2
CMAP[[3, 4, 15, 16, 17, 18, 19, 20, 21, 22]] = 3
CMAP[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] = 4
SEEDS, EPOCHS, PATIENCE, BS = 3, 200, 30, 128
MOTION_BOOST = 25.0
P = lambda n: os.path.join("data", n)


class PseudoImageSeg(nn.Module):
    def __init__(self, d):
        super().__init__()
        # translate DFS -> 3x72x128 pseudo-image
        self.proj = nn.Sequential(nn.Linear(d, 32 * 9 * 16), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1))  # [3, 72, 128]
        r18 = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(r18.conv1, r18.bn1, r18.relu, r18.maxpool,
                                  r18.layer1)  # 3x72x128 -> 64x18x32, FROZEN
        for p_ in self.stem.parameters():
            p_.requires_grad = False
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 5, 1))  # -> 5x36x64

    def forward(self, x):
        img = self.up(self.proj(x).view(-1, 32, 9, 16))
        self.stem.eval()
        return self.head(self.stem(img))


def main():
    ctr, atr = load_csi(P("A_csi.npz"), MAC)
    cho, aho = load_csi(P("demo_csi.npz"), MAC)
    dptr = np.load(P("A_dp.npz"))["dp"]
    dpho = np.load(P("demo_dp.npz"))["dp"]
    ttr = np.load(P("A_Y.npz"), allow_pickle=True)["label_ts"].astype(np.float64)
    tho = np.load(P("demo_frame_ts.npy")).astype(np.float64)
    m = min(len(ttr), len(dptr)); ttr, dptr = ttr[:m], dptr[:m]
    m = min(len(tho), len(dpho)); tho, dpho = tho[:m], dpho[:m]

    ktr = valid_mask(ctr, ttr, margin=0.35)
    kho = valid_mask(cho, tho, margin=0.35)
    XA = dfs_features(ctr, atr, ttr[ktr])
    XJ = jittered_features(ctr, atr, ttr[ktr])
    XD = dfs_features(cho, aho, tho[kho])
    TA = CMAP[dptr[ktr][:, ::10, ::10]]
    TD = CMAP[dpho[kho][:, ::10, ::10]]

    ntr = int(0.9 * len(XA))
    fm, fs = XA[:ntr].mean(0), XA[:ntr].std(0) + 1e-8
    cE, aE = load_csi(P("A_empty_csi.npz"), MAC)
    tsE = np.arange(cE[0] + W_SEC + 0.5, cE[-1] - 0.5, 0.1)
    XE = dfs_features(cE, aE, tsE)
    nEtr = int(0.8 * len(XE))

    Xz = np.concatenate([XA[:ntr], XE[:nEtr], XA[ntr:]])
    XJz = np.concatenate([XJ[:, :ntr], np.stack([XE[:nEtr]] * XJ.shape[0]), XJ[:, ntr:]], 1)
    Tz = np.concatenate([TA[:ntr], np.zeros((nEtr, GH, GW), np.int64), TA[ntr:]])
    ntr2 = ntr + nEtr
    Xz, XJz = (Xz - fm) / fs, (XJz - fm) / fs
    tt = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    Xtr, XJtr = tt(Xz[:ntr2]), tt(XJz[:, :ntr2])
    Xv, Tv = tt(Xz[ntr2:]), Tz[ntr2:]
    Ttr = torch.tensor(Tz[:ntr2], device=DEV)
    XDt, XEt = tt((XD - fm) / fs), tt((XE[nEtr:] - fm) / fs)

    cnt = np.bincount(Tz[:ntr2].ravel(), minlength=5).astype(np.float32)
    w = np.clip(cnt.sum() / (5 * cnt + 1), 0.3, 5)
    cw = torch.tensor(w / w.mean(), device=DEV, dtype=torch.float32)
    modal0 = np.zeros((GH, GW), np.int64)
    for i in range(GH):
        for j in range(GW):
            v, c = np.unique(TA[:ntr, i, j], return_counts=True)
            modal0[i, j] = v[c.argmax()]
    motion = torch.tensor((Tz[:ntr2] != modal0[None]).astype(np.float32), device=DEV)
    print(f"e3 train {ntr}+{nEtr}e, holdout {len(XDt)}", flush=True)

    probs_ho, probs_e = 0, 0
    for sd in range(SEEDS):
        torch.manual_seed(sd)
        net = PseudoImageSeg(XA.shape[1]).to(DEV)
        opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad],
                                lr=1e-3, weight_decay=1e-2)
        best, bad, state = 1e9, 0, None
        for _ in range(EPOCHS):
            net.train()
            for _ in range(0, ntr2, BS):
                idx = torch.randperm(ntr2, device=DEV)[:BS]
                src = torch.randint(XJtr.shape[0] + 1, (len(idx),), device=DEV)
                xb = Xtr[idx].clone()
                for j in range(XJtr.shape[0]):
                    mm = src == (j + 1)
                    if mm.any():
                        xb[mm] = XJtr[j][idx[mm]]
                xb = xb + 0.05 * torch.randn_like(xb)
                xb = xb * (1.0 + 0.15 * torch.randn(len(idx), 1, device=DEV))
                xb = xb + 0.10 * torch.randn_like(xb) * torch.rand(len(idx), 1, device=DEV)
                ce = F.cross_entropy(net(xb), Ttr[idx], weight=cw, reduction="none")
                pw = 1.0 + (MOTION_BOOST - 1.0) * motion[idx]
                L = (ce * pw).sum() / pw.sum()
                opt.zero_grad(); L.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                e = float(torch.cat([net(Xv[s:s + 256]).argmax(1)
                                     for s in range(0, len(Xv), 256)]).cpu().numpy().__ne__(Tv).mean())
            if e < best:
                best, bad, state = e, 0, {k: v2.clone() for k, v2 in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            probs_ho = probs_ho + F.softmax(torch.cat(
                [net(XDt[s:s + 256]) for s in range(0, len(XDt), 256)]), 1)
            probs_e = probs_e + F.softmax(torch.cat(
                [net(XEt[s:s + 256]) for s in range(0, len(XEt), 256)]), 1)
        print(f"seed {sd} done", flush=True)

    pred = (probs_ho / SEEDS).argmax(1).cpu().numpy()
    predE = (probs_e / SEEDS).argmax(1).cpu().numpy()
    rep = seg_report(pred, TD)
    rep["empty_fg_frac"] = float((predE > 0).mean())
    json.dump(rep, open("cmu_e3_report.json", "w"), indent=1)
    for k, v in rep.items():
        print(f"{k:20s} {v:.4f}")


if __name__ == "__main__":
    main()

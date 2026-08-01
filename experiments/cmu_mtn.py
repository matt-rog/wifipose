#!/usr/bin/env python3
"""CMU DensePose-From-WiFi adaptations on our single-antenna DFS data.

E1: shared multi-task trunk (seg + 3D pose + 2D kp), their keypoint-branch idea
E2: deeper modality-translation-style encoder-decoder for seg
Run from ~/wifipose:  python3 cmu_mtn.py --exp e1|e2
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/wifipose"))
from wifipose.csi import load_csi
from wifipose.dfs import dfs_features, jittered_features, valid_mask, W_SEC
from wifipose.metrics import seg_report, pose_report, constant_baseline_mpjpe
from wifipose.project import smpl_keypoints_2d

DEV = torch.device("cuda")
MAC = "0e:d7:a0:26:9d:74"
GH, GW = 36, 64
CMAP = np.zeros(25, np.int64)
CMAP[[1, 2]] = 1; CMAP[[23, 24]] = 2
CMAP[[3, 4, 15, 16, 17, 18, 19, 20, 21, 22]] = 3
CMAP[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] = 4
SEEDS, EPOCHS, PATIENCE, BS = 3, 200, 30, 128
MOTION_BOOST = 25.0
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
P = lambda n: os.path.join("data", n)


def rel(kp):
    xy, s = kp[:, :, :2].astype(np.float32), kp[:, :, 2]
    root = 0.5 * (xy[:, L_HIP] + xy[:, R_HIP])
    torso = np.linalg.norm(0.5 * (xy[:, L_SHO] + xy[:, R_SHO]) - root, axis=1) + 1e-6
    return (xy - root[:, None]) / torso[:, None, None], s


class Trunk(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.4),
                               nn.Linear(h, h), nn.GELU(), nn.Dropout(0.4))

    def forward(self, x):
        return self.t(x)


class SegHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(h, 64 * 9 * 16), nn.GELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 5, 1))

    def forward(self, z):
        return self.dec(self.proj(z).view(-1, 64, 9, 16))


class MTN(nn.Module):
    """E1: shared trunk, three heads."""
    def __init__(self, d):
        super().__init__()
        self.trunk = Trunk(d)
        self.seg = SegHead(256)
        self.pose = nn.Linear(256, 72)
        self.kp = nn.Linear(256, 34)

    def forward(self, x):
        z = self.trunk(x)
        return self.seg(z), self.pose(z), self.kp(z)


class DeepSeg(nn.Module):
    """E2: translation-style spatial encoder-decoder (their fig 4 shape)."""
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, 32 * 9 * 16), nn.GELU())
        self.enc = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 5, 1))

    def forward(self, x):
        return self.dec(self.enc(self.proj(x).view(-1, 32, 9, 16)))


def aug(xb, XJtr, idx):
    src = torch.randint(XJtr.shape[0] + 1, (len(idx),), device=DEV)
    for j in range(XJtr.shape[0]):
        mm = src == (j + 1)
        if mm.any():
            xb[mm] = XJtr[j][idx[mm]]
    xb = xb + 0.05 * torch.randn_like(xb)
    xb = xb * (1.0 + 0.15 * torch.randn(len(idx), 1, device=DEV))
    xb = xb + 0.10 * torch.randn_like(xb) * torch.rand(len(idx), 1, device=DEV)
    return xb


def main(a):
    ctr, atr = load_csi(P("A_csi.npz"), MAC)
    cho, aho = load_csi(P("demo_csi.npz"), MAC)
    AY = np.load(P("A_Y.npz"), allow_pickle=True)
    DY = np.load(P("demo_Y.npz"), allow_pickle=True)
    dptr = np.load(P("A_dp.npz"))["dp"]
    dpho = np.load(P("demo_dp.npz"))["dp"]
    ttr = AY["label_ts"].astype(np.float64)
    m = min(len(ttr), len(dptr)); ttr, dptr = ttr[:m], dptr[:m]
    Jtr_all = AY["J_canon"][:m].astype(np.float32)
    kptr_all = smpl_keypoints_2d(AY["J_canon"].astype(np.float32), AY["R_can"],
                                 AY["pelvis"], float(AY["height"]))[:m]

    ktr = valid_mask(ctr, ttr, margin=0.35)
    XA = dfs_features(ctr, atr, ttr[ktr])
    XJ = jittered_features(ctr, atr, ttr[ktr])
    TA = CMAP[dptr[ktr][:, ::10, ::10]]
    Jtr = Jtr_all[ktr]
    Ktr, Str = rel(kptr_all[ktr])

    # seg holdout on frame_ts; pose/kp holdout on demo_Y label_ts
    tho_f = np.load(P("demo_frame_ts.npy")).astype(np.float64)
    m2 = min(len(tho_f), len(dpho)); tho_f, dpho = tho_f[:m2], dpho[:m2]
    kho_f = valid_mask(cho, tho_f, margin=0.35)
    XD_seg = dfs_features(cho, aho, tho_f[kho_f])
    TD = CMAP[dpho[kho_f][:, ::10, ::10]]
    tho_p = DY["label_ts"].astype(np.float64)
    kho_p = valid_mask(cho, tho_p)
    XD_pose = dfs_features(cho, aho, tho_p[kho_p])
    Jho = DY["J_canon"][kho_p].astype(np.float32)

    ntr = int(0.9 * len(XA))
    fm, fs = XA[:ntr].mean(0), XA[:ntr].std(0) + 1e-8

    cE, aE = load_csi(P("A_empty_csi.npz"), MAC)
    tsE = np.arange(cE[0] + W_SEC + 0.5, cE[-1] - 0.5, 0.1)
    XE = dfs_features(cE, aE, tsE)
    nEtr = int(0.8 * len(XE))

    Xz = np.concatenate([XA[:ntr], XE[:nEtr], XA[ntr:]])
    XJz = np.concatenate([XJ[:, :ntr], np.stack([XE[:nEtr]] * XJ.shape[0]), XJ[:, ntr:]], 1)
    Tz = np.concatenate([TA[:ntr], np.zeros((nEtr, GH, GW), np.int64), TA[ntr:]])
    Yp = Jtr.reshape(-1, 72)
    ym, ys = Yp[:ntr].mean(0), Yp[:ntr].std(0) + 1e-8
    Yk = Ktr.reshape(-1, 34)
    km, ks = Yk[:ntr].mean(0), Yk[:ntr].std(0) + 1e-8
    Ypz = np.concatenate([(Yp[:ntr] - ym) / ys, np.zeros((nEtr, 72), np.float32),
                          (Yp[ntr:] - ym) / ys]).astype(np.float32)
    Ykz = np.concatenate([(Yk[:ntr] - km) / ks, np.zeros((nEtr, 34), np.float32),
                          (Yk[ntr:] - km) / ks]).astype(np.float32)
    Vz = np.concatenate([np.ones(ntr), np.zeros(nEtr), np.ones(len(XA) - ntr)]).astype(np.float32)
    ntr2 = ntr + nEtr
    Xz, XJz = (Xz - fm) / fs, (XJz - fm) / fs

    tt = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    Xtr, XJtr = tt(Xz[:ntr2]), tt(XJz[:, :ntr2])
    Xv, Tv = tt(Xz[ntr2:]), Tz[ntr2:]
    Ttr = torch.tensor(Tz[:ntr2], device=DEV)
    Yptr, Yktr, Vtr = tt(Ypz[:ntr2]), tt(Ykz[:ntr2]), tt(Vz[:ntr2])
    Ypv = tt(((Yp[ntr:] - ym) / ys))
    XDs = tt((XD_seg - fm) / fs)
    XDp = tt((XD_pose - fm) / fs)
    XEt = tt((XE[nEtr:] - fm) / fs)

    cnt = np.bincount(Tz[:ntr2].ravel(), minlength=5).astype(np.float32)
    w = np.clip(cnt.sum() / (5 * cnt + 1), 0.3, 5)
    cw = torch.tensor(w / w.mean(), device=DEV, dtype=torch.float32)
    modal0 = np.zeros((GH, GW), np.int64)
    for i in range(GH):
        for j in range(GW):
            v, c = np.unique(TA[:ntr, i, j], return_counts=True)
            modal0[i, j] = v[c.argmax()]
    motion = torch.tensor((Tz[:ntr2] != modal0[None]).astype(np.float32), device=DEV)
    hub = nn.SmoothL1Loss(beta=0.1, reduction="none")
    print(f"exp={a.exp} train {ntr}+{nEtr}e, seg-ho {len(XDs)}, pose-ho {len(XDp)}", flush=True)

    probs_ho, probs_e, pose_preds = 0, 0, []
    for sd in range(SEEDS):
        torch.manual_seed(sd)
        net = (MTN if a.exp == "e1" else DeepSeg)(XA.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        best, bad, state = 1e9, 0, None
        for _ in range(EPOCHS):
            net.train()
            for _ in range(0, ntr2, BS):
                idx = torch.randperm(ntr2, device=DEV)[:BS]
                xb = aug(Xtr[idx].clone(), XJtr, idx)
                if a.exp == "e1":
                    sg, pz, kz = net(xb)
                else:
                    sg = net(xb)
                ce = F.cross_entropy(sg, Ttr[idx], weight=cw, reduction="none")
                pw = 1.0 + (MOTION_BOOST - 1.0) * motion[idx]
                L = (ce * pw).sum() / pw.sum()
                if a.exp == "e1":
                    v = Vtr[idx][:, None]
                    L = L + 0.3 * (hub(pz, Yptr[idx]) * v).sum() / (v.sum() * 72 + 1e-6)
                    L = L + 0.3 * (hub(kz, Yktr[idx]) * v).sum() / (v.sum() * 34 + 1e-6)
                opt.zero_grad(); L.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                if a.exp == "e1":
                    sgv, pzv, _ = net(Xv)
                    e = float((sgv.argmax(1).cpu().numpy() != Tv).mean()) \
                        + 0.1 * float(((pzv - Ypv) ** 2).mean())
                else:
                    e = float((net(Xv).argmax(1).cpu().numpy() != Tv).mean())
            if e < best:
                best, bad, state = e, 0, {k: v2.clone() for k, v2 in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            if a.exp == "e1":
                probs_ho = probs_ho + F.softmax(torch.cat(
                    [net(XDs[s:s + 256])[0] for s in range(0, len(XDs), 256)]), 1)
                probs_e = probs_e + F.softmax(torch.cat(
                    [net(XEt[s:s + 256])[0] for s in range(0, len(XEt), 256)]), 1)
                pose_preds.append((net(XDp)[1].cpu().numpy() * ys + ym).reshape(-1, 24, 3))
            else:
                probs_ho = probs_ho + F.softmax(torch.cat(
                    [net(XDs[s:s + 256]) for s in range(0, len(XDs), 256)]), 1)
                probs_e = probs_e + F.softmax(torch.cat(
                    [net(XEt[s:s + 256]) for s in range(0, len(XEt), 256)]), 1)
        print(f"seed {sd} done", flush=True)

    pred = (probs_ho / SEEDS).argmax(1).cpu().numpy()
    predE = (probs_e / SEEDS).argmax(1).cpu().numpy()
    rep = seg_report(pred, TD)
    rep["empty_fg_frac"] = float((predE > 0).mean())
    if a.exp == "e1":
        Jp = np.mean(pose_preds, 0)
        pr = pose_report(Jp, Jho)
        rep.update({f"pose_{k}": v for k, v in pr.items()})
        rep["pose_const_mpjpe"] = constant_baseline_mpjpe(Jtr[:ntr], Jho[:min(len(Jp), len(Jho))])
    json.dump(rep, open(f"cmu_{a.exp}_report.json", "w"), indent=1)
    for k, v in rep.items():
        print(f"{k:20s} {v:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["e1", "e2"])
    main(ap.parse_args())

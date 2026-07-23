#!/usr/bin/env python3
"""RuView DensePose baseline: their DensePoseHead (24-part, their segmentation
loss) fed by a linear CSI projector, raw-amplitude input, on our data and eval.

python ruview/run_densepose.py --mac <bssid> --data data
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RuView", "examples", "through-wall"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RuView", "archive", "v1", "src", "models"))
from wifipose.csi import load_csi
from wifipose.metrics import seg_report
from densepose_head import DensePoseHead
from run_skeleton import amp_features, WIN

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GH, GW = 36, 64
EPOCHS, PATIENCE, BS = 200, 30, 128


def main(a):
    p = lambda n: os.path.join(a.data, n)
    ctr, atr = load_csi(p(f"{a.train}_csi.npz"), a.mac)
    cho, aho = load_csi(p(f"{a.holdout}_csi.npz"), a.mac)
    dptr = np.load(p(f"{a.train}_dp.npz"))["dp"]
    dpho = np.load(p(f"{a.holdout}_dp.npz"))["dp"]
    ttr = np.load(p(f"{a.train}_Y.npz"), allow_pickle=True)["label_ts"].astype(np.float64)
    tho = np.load(p(f"{a.holdout}_frame_ts.npy")).astype(np.float64)
    m = min(len(ttr), len(dptr)); ttr, dptr = ttr[:m], dptr[:m]
    m = min(len(tho), len(dpho)); tho, dpho = tho[:m], dpho[:m]
    ktr = (ttr >= ctr[0] + WIN) & (ttr <= ctr[-1])
    kho = (tho >= cho[0] + WIN) & (tho <= cho[-1])
    XA = amp_features(ctr, atr, ttr[ktr])
    XD = amp_features(cho, aho, tho[kho])
    TA = dptr[ktr][:, ::10, ::10].astype(np.int64)  # 24-part targets, their design
    TD = dpho[kho][:, ::10, ::10].astype(np.int64)

    ntr = int(0.9 * len(XA))
    fm, fs = XA[:ntr].mean(0), XA[:ntr].std(0) + 1e-8
    tt = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    Xtr, Xv = tt(((XA - fm) / fs)[:ntr]), tt(((XA - fm) / fs)[ntr:])
    Ttr, Tv = torch.tensor(TA[:ntr], device=DEV), TA[ntr:]
    XDt = tt((XD - fm) / fs)

    torch.manual_seed(0)
    proj = nn.Sequential(nn.Linear(XA.shape[1], 256 * 8 * 8), nn.GELU()).to(DEV)
    head = DensePoseHead(dict(input_channels=256, num_body_parts=24,
                              num_uv_coordinates=2, hidden_channels=[128, 64],
                              kernel_size=3, padding=1, dropout_rate=0.1,
                              use_fpn=False, output_stride=4)).to(DEV)

    def forward(xb):
        seg = head(proj(xb).view(-1, 256, 8, 8))["segmentation"]
        return F.interpolate(seg, size=(GH, GW), mode="bilinear", align_corners=False)

    opt = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-2)
    best, bad, state = 1e9, 0, None
    for _ in range(EPOCHS):
        proj.train(); head.train()
        for s in range(0, ntr, BS):
            idx = torch.randperm(ntr, device=DEV)[:BS]
            L = head.compute_segmentation_loss(forward(Xtr[idx]), Ttr[idx])
            opt.zero_grad(); L.backward(); opt.step()
        proj.eval(); head.eval()
        with torch.no_grad():
            e = float((forward(Xv).argmax(1).cpu().numpy() != Tv).mean())
        if e < best:
            best, bad = e, 0
            state = ({k: v.clone() for k, v in proj.state_dict().items()},
                     {k: v.clone() for k, v in head.state_dict().items()})
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    proj.load_state_dict(state[0]); head.load_state_dict(state[1])
    proj.eval(); head.eval()
    with torch.no_grad():
        pred = torch.cat([forward(XDt[s:s + 256]) for s in range(0, len(XDt), 256)]) \
            .argmax(1).cpu().numpy()
    rep = seg_report(pred, TD, nclass=25)
    modal = np.zeros((GH, GW), np.int64)
    for i in range(GH):
        for j in range(GW):
            v, c = np.unique(TA[:ntr, i, j], return_counts=True)
            modal[i, j] = v[c.argmax()]
    rep["static_fg_iou"] = seg_report(np.tile(modal, (len(TD), 1, 1)), TD, nclass=25)["fg_iou"]
    np.save(p("ruview_densepose_pred.npy"), pred)
    json.dump(rep, open(p("ruview_densepose_report.json"), "w"), indent=1)
    for k, v in rep.items():
        print(f"{k:16s} {v:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train")
    ap.add_argument("--holdout", default="holdout")
    ap.add_argument("--mac", required=True)
    ap.add_argument("--data", default="data")
    main(ap.parse_args())

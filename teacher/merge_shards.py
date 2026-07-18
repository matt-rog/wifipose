#!/usr/bin/env python3
"""Merge sharded CoMotion .pt outputs into ONE single-track .pt.

CoMotion was run in parallel over frame-range shards (one per GPU). Within each
shard `frame_idx` is 0-based RELATIVE to that shard's --start-frame, and track
IDs are per-shard (shard A's id 2 != shard B's id 2). So per shard we: pick the
dominant track (argmax bincount), offset its frame_idx by the shard start, and
relabel it to a common id 0. Then concat + sort by global frame_idx.

  venv/bin/python code/comotion_merge.py --out data/A.pt \
      data/shard0/A_dec.pt:0 data/shard1/A_dec.pt:2800 data/shard2/A_dec.pt:5600
"""
import argparse
import numpy as np
import torch


def main(a):
    ids, pose, trans, betas, fidx = [], [], [], [], []
    for spec in a.shards:
        path, S = spec.rsplit(":", 1)
        S = int(S)
        d = torch.load(path, map_location="cpu", weights_only=True)
        sid = d["id"].numpy().astype(int)
        dom = np.bincount(sid - sid.min()).argmax() + sid.min()
        m = sid == dom
        gfi = d["frame_idx"].numpy().astype(np.int64)[m] + S
        ids.append(np.zeros(int(m.sum()), np.int64))
        pose.append(d["pose"].numpy()[m])
        trans.append(d["trans"].numpy()[m])
        betas.append(d["betas"].numpy()[m])
        fidx.append(gfi)
        print(f"  {path}: dom track {dom}, {int(m.sum())} frames, "
              f"global {gfi.min()}..{gfi.max()}")
    fi = np.concatenate(fidx)
    order = np.argsort(fi, kind="stable")
    out = dict(
        id=torch.from_numpy(np.concatenate(ids)[order]),
        pose=torch.from_numpy(np.concatenate(pose)[order].astype(np.float32)),
        trans=torch.from_numpy(np.concatenate(trans)[order].astype(np.float32)),
        betas=torch.from_numpy(np.concatenate(betas)[order].astype(np.float32)),
        frame_idx=torch.from_numpy(fi[order]),
    )
    torch.save(out, a.out)
    # gap report
    u = np.unique(fi)
    gaps = np.diff(u)
    print(f"merged -> {a.out}: {len(fi)} frames, {len(u)} unique idx, "
          f"span {u.min()}..{u.max()}, max gap {gaps.max() if len(gaps) else 0} "
          f"({(gaps > 1).sum()} gaps >1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+", help="path.pt:start_frame per shard")
    main(ap.parse_args())

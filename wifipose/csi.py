"""Nexmon CSI loading (BCM43455c0, 80 MHz). Wire format: seemoo-lab/nexmon_csi."""
import numpy as np

NULL = set(range(0, 6)) | {127, 128, 129} | set(range(251, 256))
KEEP = [i for i in range(256) if i not in NULL]  # 242 live subcarriers
NEXMON_HDR = 18


N_SUB = 114


def load_csi(npz_path, mac):
    """Return (ts [M], amp [M,114] L1-normalized), filtered to one TX MAC and
    its dominant frame type, downsampled 242 -> 114 subcarriers. Phase is
    discarded (unusable on a single antenna)."""
    d = np.load(npz_path)
    want = np.array([int(x, 16) for x in mac.split(":")], np.uint8)
    sel = (d["csi_mac"] == want).all(1)
    v, c = np.unique(d["csi_fctl"][sel], return_counts=True)
    sel &= d["csi_fctl"] == int(v[c.argmax()])
    raw = d["csi_raw"][sel].astype(np.float32)[:, KEEP, :]
    ts = d["csi_ts"][sel]
    order = np.argsort(ts, kind="stable")
    amp = np.hypot(raw[order, :, 0], raw[order, :, 1])
    amp /= amp.mean(1, keepdims=True) + 1e-6
    idx = np.linspace(0, len(KEEP) - 1, N_SUB)
    amp = np.stack([np.interp(idx, np.arange(len(KEEP)), r) for r in amp])
    return ts[order].astype(np.float64), amp.astype(np.float32)

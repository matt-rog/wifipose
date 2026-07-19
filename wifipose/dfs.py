"""Single-antenna Doppler (DFS) features. See README for the derivation
(CARM 2015 / SHARP 2023 / Widar3.0 single-link slice)."""
import numpy as np

W_SEC, T_FFT = 2.0, 128
JITTER_DELTAS = (-0.3, -0.15, 0.15, 0.3)


def _interp(tw, A, grid):
    idx = np.searchsorted(tw, grid).clip(1, len(tw) - 1)
    t0, t1 = tw[idx - 1], tw[idx]
    w = ((grid - t0) / (t1 - t0 + 1e-9)).clip(0, 1)
    return A[idx - 1] * (1 - w)[:, None] + A[idx] * w[:, None]


def dfs_features(cts, amp, at_ts, W=W_SEC, T=T_FFT):
    """Per query time: trailing W-second window -> uniform grid -> mean-subtract
    -> Hann -> per-subcarrier |rFFT|, averaged over all subcarriers + 3 bands."""
    K = amp.shape[1]
    bands = [slice(0, K // 3), slice(K // 3, 2 * K // 3), slice(2 * K // 3, K)]
    hann = np.hanning(T)
    out = []
    for t in at_ts:
        hi = np.searchsorted(cts, t)
        lo = np.searchsorted(cts, t - W)
        tw, aw = cts[lo:hi], amp[lo:hi]
        if len(tw) >= 8:
            U = _interp(tw, aw, np.linspace(t - W, t, T))
            U = U - U.mean(0, keepdims=True)
            mag = np.abs(np.fft.rfft(U * hann[:, None], axis=0))
            f = np.concatenate([mag.mean(1)] + [mag[:, b].mean(1) for b in bands])
        else:
            f = np.zeros((T // 2 + 1) * 4)
        out.append(f.astype(np.float32))
    return np.array(out, np.float32)


def valid_mask(cts, at_ts, W=W_SEC, margin=0.0):
    return ((np.searchsorted(cts, at_ts) - np.searchsorted(cts, at_ts - W) >= 8)
            & (at_ts >= cts[0] + W + margin) & (at_ts <= cts[-1] - margin))


def jittered_features(cts, amp, at_ts, deltas=JITTER_DELTAS):
    """Time-shifted feature copies for augmentation. [len(deltas), N, d]"""
    return np.stack([dfs_features(cts, amp, at_ts + dt) for dt in deltas])

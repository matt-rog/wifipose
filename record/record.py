#!/usr/bin/env python3
"""Synchronized camera + CSI recorder. Both streams stamped with
time.monotonic() at arrival; offline pairing = timestamp lookup + the constant
offset from sync_offset.py.

python record/record.py --prefix A --secs 840 --cam-device 4 --exposure 350 --gain 255
"""
import argparse, signal, socket, subprocess, threading, time
import numpy as np, cv2

buf = {k: [] for k in ("ts", "raw", "seq", "mac", "chanspec", "rssi", "fctl")}
stop = threading.Event()
signal.signal(signal.SIGINT, lambda *a: stop.set())
signal.signal(signal.SIGTERM, lambda *a: stop.set())


def csi_thread(port):
    """Nexmon CSI UDP packets (seemoo-lab/nexmon_csi, 18-byte header)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.settimeout(1.0)
    while not stop.is_set():
        try:
            p = s.recvfrom(4096)[0]
        except socket.timeout:
            continue
        ts = time.monotonic()
        raw = np.frombuffer(p[18:], np.int16)
        raw = raw[:(raw.size // 2) * 2].reshape(-1, 2)
        r = np.zeros((256, 2), np.int16)
        r[:min(256, len(raw))] = raw[:256]
        buf["ts"].append(ts)
        buf["raw"].append(r)
        buf["rssi"].append(np.frombuffer(p[2:3], np.int8)[0])
        buf["fctl"].append(p[3])
        buf["mac"].append(np.frombuffer(p[4:10], np.uint8).copy())
        buf["seq"].append(int.from_bytes(p[10:12], "little"))
        buf["chanspec"].append(int.from_bytes(p[14:16], "little"))


def lock_camera(dev, exposure, gain):
    """Manual exposure/gain, AF+AWB off. exposure=0 leaves auto."""
    if not exposure:
        return
    for ctrl in (["auto_exposure=1", f"exposure_time_absolute={exposure}",
                  "focus_automatic_continuous=0", "white_balance_automatic=0"]
                 + ([f"gain={gain}"] if gain else [])):
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{dev}", "--set-ctrl", ctrl],
                       capture_output=True)


def main(a):
    threading.Thread(target=csi_thread, args=(a.port,), daemon=True).start()
    time.sleep(1.0)
    lock_camera(a.cam_device, a.exposure, a.gain)
    cap = cv2.VideoCapture(a.cam_device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vw = cv2.VideoWriter(f"{a.prefix}_video.avi", cv2.VideoWriter_fourcc(*"MJPG"),
                         30, (1280, 720))
    vts, t0 = [], time.monotonic()
    while not stop.is_set() and time.monotonic() - t0 < a.secs:
        ok, frame = cap.read()
        ts = time.monotonic()
        if not ok:
            break
        vw.write(frame)
        vts.append(ts)
        if len(vts) % 300 == 0:
            rate = sum(1 for x in buf["ts"][-2000:] if ts - x < 1.0)
            print(f"t={ts - t0:5.0f}s frames={len(vts)} csi={rate}/s", flush=True)
    stop.set()
    cap.release()
    vw.release()

    M = min(len(buf["ts"]), len(buf["raw"]))
    np.save(f"{a.prefix}_frame_ts.npy", np.array(vts))
    np.savez(f"{a.prefix}_csi.npz",
             csi_ts=np.array(buf["ts"][:M]),
             csi_raw=np.array(buf["raw"][:M]),
             csi_seq=np.array(buf["seq"][:M], np.uint16),
             csi_mac=np.array(buf["mac"][:M], np.uint8),
             csi_chanspec=np.array(buf["chanspec"][:M], np.uint16),
             csi_rssi=np.array(buf["rssi"][:M], np.int8),
             csi_fctl=np.array(buf["fctl"][:M], np.uint8))

    cts = np.array(buf["ts"][:M])
    dur = vts[-1] - vts[0] if vts else 0
    print(f"video {len(vts)} frames ({len(vts) / max(dur, 1e-9):.1f} fps), "
          f"CSI {M} pkts ({M / max(dur, 1e-9):.0f}/s)")
    craw = np.array(buf["raw"][:M], np.float32)
    nz = float((np.abs(craw).sum(axis=(1, 2)) > 0).mean())
    print(f"raw complex nonzero: {100 * nz:.0f}%" + ("" if nz > 0.95 else "  WARNING: truncated"))
    macs, cc = np.unique(np.array(buf["mac"][:M], np.uint8), axis=0, return_counts=True)
    dom = macs[cc.argmax()]
    print(f"dominant TX {':'.join(f'{b:02x}' for b in dom)}: {100 * cc.max() / cc.sum():.0f}%")
    lo = np.searchsorted(cts, np.array(vts) - 0.5)
    hi = np.searchsorted(cts, vts)
    print(f"frames with >=10 CSI pkts in 0.5s: {100 * float((hi - lo >= 10).mean()):.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--secs", type=int, default=840)
    ap.add_argument("--port", type=int, default=5501)
    ap.add_argument("--cam-device", type=int, default=0)
    ap.add_argument("--exposure", type=int, default=0, help="~100us units, 0=auto")
    ap.add_argument("--gain", type=int, default=0)
    main(ap.parse_args())

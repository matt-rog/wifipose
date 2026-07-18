#!/bin/bash
# One-command Pi CSI bring-up after power-on. Run from the laptop.
#   ./pi_bringup.sh [pi_ip] [laptop_ip] [chanspec]
# AP "m&m" 5 GHz moved to primary channel 48 (2026-07-17); 48/80 = same 80 MHz
# block (center 42) as the old 44/80.
set -e
PI=${1:-192.168.8.106}
LAPTOP=${2:-$(ip -4 route get 192.168.8.1 | grep -oP 'src \K\S+')}
CHAN=${3:-48/80}

echo "== waiting for Pi at $PI =="
until ping -c1 -W2 "$PI" >/dev/null 2>&1; do sleep 2; printf .; done; echo " up"

SSH="ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new pi@$PI"
echo "== enabling CSI on chanspec $CHAN =="
$SSH "picsi up $CHAN && sudo sysctl -w net.ipv4.conf.all.rp_filter=0 net.ipv4.conf.wlan0.rp_filter=0"

echo "== starting 5500 -> $LAPTOP:5501 relay =="
$SSH "pkill -f 'csi_rela[y]' 2>/dev/null; cat > /tmp/csi_relay.py <<'EOF'
import socket
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
rx.bind((\"0.0.0.0\", 5500))
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
while True:
    tx.sendto(rx.recv(4096), (\"$LAPTOP\", 5501))
EOF
nohup python3 /tmp/csi_relay.py > /tmp/csi_relay.log 2>&1 & echo relay pid \$!"

echo "== verifying packet rate on laptop:5501 (5s) =="
timeout 6 python3 - <<EOF
import socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 5501)); s.settimeout(6)
t0 = time.time(); n = 0
try:
    while time.time() - t0 < 5:
        s.recvfrom(4096); n += 1
except socket.timeout:
    pass
rate = n / 5
print(f"CSI rate: {rate:.0f} pkt/s ->", "OK" if rate > 50 else
      "LOW — check AP channel (nmcli dev wifi list), traffic, chanspec")
EOF
echo "== done. next: python3 csi_presence.py for the live movement check =="

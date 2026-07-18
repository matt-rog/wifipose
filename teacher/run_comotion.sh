#!/usr/bin/env bash
# Run the CoMotion teacher (apple/ml-comotion) on a video, sharded across GPUs,
# and merge the dominant track. Usage: ./run_comotion.sh <video> <total_frames> <out.pt>
set -e
VIDEO=$1; TOTAL=$2; OUT=$3
PY=${PY:-python}
DEMO=${COMOTION:-ml-comotion/demo.py}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
N=$(( (TOTAL + NGPU - 1) / NGPU ))
TMP=$(mktemp -d)

for g in $(seq 0 $((NGPU - 1))); do
  S=$((g * N))
  mkdir -p "$TMP/shard$g"
  CUDA_VISIBLE_DEVICES=$g $PY $DEMO -i "$VIDEO" -o "$TMP/shard$g" -s $S -n $N \
    --skip-visualization > "$TMP/shard$g.log" 2>&1 &
done
wait

SHARDS=""
BASE=$(basename "$VIDEO"); BASE=${BASE%.*}
for g in $(seq 0 $((NGPU - 1))); do
  SHARDS="$SHARDS $TMP/shard$g/$BASE.pt:$((g * N))"
done
$PY "$(dirname "$0")/merge_shards.py" --out "$OUT" $SHARDS
rm -rf "$TMP"

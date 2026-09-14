#!/usr/bin/env bash
# ==============================================================================
# 04_generate_chunks.sh -- STEP 4 of 4
# ------------------------------------------------------------------------------
# Runs generate_chunks_avcocktail.py over the train and dev splits: slices
# each stereo session wav into overlapping WINDOW_SIZE-second chunks (stride
# STEP_SIZE seconds) and saves the VAD/VAP one-hot labels for each chunk.
#
# generate_chunks_avcocktail.py auto-discovers which stereo wavs to chunk by
# scanning <split>/session_*/central-stereo_audios/*.wav, so step 3 must
# have produced that output first.
#
# Usage:
#   ./04_generate_chunks.sh                        # full train + dev
#   PARTS="dev" SESSIONS="session_53" ./04_generate_chunks.sh   # single session
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

echo "======================================================================"
echo "STEP 4/4: generate audio chunks + VAD/VAP labels"
echo "======================================================================"
echo "window_size=${WINDOW_SIZE}s step_size=${STEP_SIZE}s future_context=${FUTURE_CONTEXT}s"
echo "output dir: $CHUNKS_OUT_DIR"

PARTS="${PARTS:-train dev}"
SESSIONS="${SESSIONS:-}"   # comma/space separated session names to restrict to (all if empty)
# normalise SESSIONS to comma-separated for the python --sessions flag
SESSIONS_CSV="$(echo "$SESSIONS" | tr ' ' ',' )"

mkdir -p "$CHUNKS_OUT_DIR"

for part in $PARTS; do
    part_dir="$SRC_AVCOCKTAIL/$part"
    if [ ! -d "$part_dir" ]; then
        echo "[4/4] WARNING: $part_dir does not exist, skipping '$part' split."
        continue
    fi

    out_prefix="$CHUNKS_OUT_DIR/$part"
    echo "[4/4] $part -> $out_prefix"

    extra_args=()
    if [ -n "$SESSIONS_CSV" ]; then
        extra_args+=(--sessions "$SESSIONS_CSV")
    fi

    conda run --no-capture-output -n "$CHUNKS_CONDA_ENV" python3 "$SCRIPT_DIR/generate_chunks_avcocktail.py" \
        --audio-dir "$part_dir" \
        --out-prefix "$out_prefix" \
        "${extra_args[@]}"
done

echo "[4/4] done."

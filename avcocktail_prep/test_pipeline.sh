#!/usr/bin/env bash
# ==============================================================================
# test_pipeline.sh
# ------------------------------------------------------------------------------
# Sanity-checks steps 3+4 of the pipeline on ONE train session and ONE dev
# session, WITHOUT touching $SRC_AVCOCKTAIL:
#   1. copies each test session's RAW inputs (metadata, labels, raw face-crop
#      tracks) into an isolated scratch root, stripping any already-generated
#      derived output (central-stereo_audios/, aligned_reset videos) so
#      process_session.py regenerates everything from scratch.
#   2. runs process_session.py (step 3) against that scratch root.
#   3. runs generate_chunks_avcocktail.py (step 4) against that scratch root,
#      restricted to the two test sessions.
#   4. compares the freshly generated stereo wav / aligned video / chunk
#      output against whatever is already sitting in $SRC_AVCOCKTAIL for that
#      same session (where such a baseline exists).
#
# Usage:
#   ./test_pipeline.sh                                   # defaults below
#   TRAIN_TEST_SESSION=session_01 DEV_TEST_SESSION=session_40 ./test_pipeline.sh
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

# session_00 (train, 447MB, 1 pair) and session_53 (dev, 2.4GB, 1 pair,
# already has "_fixed" baseline output) -- picked as the smallest sessions
# that actually exercise the 2-speaker alignment path.
TRAIN_TEST_SESSION="${TRAIN_TEST_SESSION:-session_00}"
DEV_TEST_SESSION="${DEV_TEST_SESSION:-session_53}"

TEST_ROOT="$SCRIPT_DIR/_pipeline_test"
TEST_CHUNKS_DIR="$TEST_ROOT/_chunks"

echo "======================================================================"
echo "PIPELINE TEST: train/$TRAIN_TEST_SESSION + dev/$DEV_TEST_SESSION"
echo "  source (read-only): $SRC_AVCOCKTAIL"
echo "  scratch (writable):  $TEST_ROOT"
echo "======================================================================"

# ---- 1. copy raw inputs into an isolated scratch root -----------------------
copy_raw_session() {
    local part="$1" session="$2"
    local src="$SRC_AVCOCKTAIL/$part/$session"
    local dst="$TEST_ROOT/$part/$session"

    if [ ! -d "$src" ]; then
        echo "  ERROR: $src does not exist" >&2
        exit 1
    fi

    echo "[test] copying $part/$session raw inputs -> $dst"
    mkdir -p "$dst"
    rsync -a --delete \
        --exclude 'central-stereo_audios/' \
        --exclude 'stereo_audios/' \
        "$src/" "$dst/"

    # strip any already-generated aligned/upsampled artifacts inside the
    # per-speaker crop dirs, so process_session.py regenerates them fresh.
    find "$dst/speakers" -type f \( -name "*_central_aligned_reset*" \) -delete 2>/dev/null || true
}

copy_raw_session "train" "$TRAIN_TEST_SESSION"
copy_raw_session "dev" "$DEV_TEST_SESSION"

# process_session.py needs the actual per-speaker crop videos (track_*.mp4).
# Some sessions in this local dataset snapshot only have derived artifacts
# (bbox/asd json, extracted wav) but not the raw crop mp4 itself -- most
# likely because the "*_without_central_videos.zip" used in step 1 excludes
# them, and only some sessions were separately supplemented with the raw
# video. Detect that up front per split instead of crashing mid-run.
has_crop_video() {
    local part="$1" session="$2"
    find "$TEST_ROOT/$part/$session/speakers" -name "track_*.mp4" 2>/dev/null | grep -q .
}

RUN_TRAIN=1
RUN_DEV=1
if ! has_crop_video "train" "$TRAIN_TEST_SESSION"; then
    echo
    echo "[test] WARNING: train/$TRAIN_TEST_SESSION has no track_*.mp4 crop video locally."
    echo "[test]          (checked every session in $SRC_AVCOCKTAIL/train -- none currently have one)"
    echo "[test]          process_session.py needs it to build the aligned video/audio."
    echo "[test]          Skipping the train half of this test; see the run's final summary."
    RUN_TRAIN=0
fi
if ! has_crop_video "dev" "$DEV_TEST_SESSION"; then
    echo
    echo "[test] WARNING: dev/$DEV_TEST_SESSION has no track_*.mp4 crop video locally, skipping dev half."
    RUN_DEV=0
fi

# ---- 2. run step 3 (process_session.py) against the scratch root -----------
echo
echo "[test] running process_session.py on the scratch copies..."
[ "$RUN_TRAIN" = "1" ] && python3 "$SCRIPT_DIR/process_session.py" --root "$TEST_ROOT" --part train --session "$TRAIN_TEST_SESSION"
[ "$RUN_DEV" = "1" ] && python3 "$SCRIPT_DIR/process_session.py" --root "$TEST_ROOT" --part dev --session "$DEV_TEST_SESSION"

# ---- 3. run step 4 (generate_chunks_avcocktail.py) against the scratch root -
echo
echo "[test] running generate_chunks_avcocktail.py on the scratch copies..."
mkdir -p "$TEST_CHUNKS_DIR"
if [ "$RUN_TRAIN" = "1" ]; then
    conda run --no-capture-output -n "$CHUNKS_CONDA_ENV" python3 "$SCRIPT_DIR/generate_chunks_avcocktail.py" \
        --audio-dir "$TEST_ROOT/train" --out-prefix "$TEST_CHUNKS_DIR/train" --sessions "$TRAIN_TEST_SESSION"
fi
if [ "$RUN_DEV" = "1" ]; then
    conda run --no-capture-output -n "$CHUNKS_CONDA_ENV" python3 "$SCRIPT_DIR/generate_chunks_avcocktail.py" \
        --audio-dir "$TEST_ROOT/dev" --out-prefix "$TEST_CHUNKS_DIR/dev" --sessions "$DEV_TEST_SESSION"
fi

# ---- 4. compare against $SRC_AVCOCKTAIL ------------------------------------
echo
echo "======================================================================"
echo "COMPARISON vs current data in $SRC_AVCOCKTAIL"
echo "======================================================================"
COMPARE_ARGS=(--src-root "$SRC_AVCOCKTAIL" --test-root "$TEST_ROOT")
[ "$RUN_TRAIN" = "1" ] && COMPARE_ARGS+=(--train-session "$TRAIN_TEST_SESSION")
[ "$RUN_DEV" = "1" ] && COMPARE_ARGS+=(--dev-session "$DEV_TEST_SESSION")
conda run --no-capture-output -n "$CHUNKS_CONDA_ENV" python3 "$SCRIPT_DIR/compare_test_outputs.py" "${COMPARE_ARGS[@]}"

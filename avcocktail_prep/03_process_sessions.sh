#!/usr/bin/env bash
# ==============================================================================
# 03_process_sessions.sh -- STEP 3 of 4
# ------------------------------------------------------------------------------
# Runs process_session.py over every session in both the train and dev
# splits. For each session, process_session.py:
#   1. aligns each speaker's central-camera face-crop track(s) onto the
#      shared session timeline and cuts to the whistle-aligned
#      [uem_start, uem_end] window (build_aligned_reset)
#   2. upsamples that video to 30fps and extracts its mono wav
#   3. merges each dyadic pair's two mono wavs into one stereo wav, written
#      to <session>/central-stereo_audios/<spkA>-<spkB>-30fps_fixed.wav
#
# Sessions whose interlocutor clusters never pair up 1-on-1 (e.g. a single
# 3-person group) are skipped by process_session.py itself and simply
# produce no stereo output.
#
# Usage:
#   ./03_process_sessions.sh                # process every train+dev session
#   FORCE=1 ./03_process_sessions.sh         # re-process even if output exists
#   PARTS="dev" SESSIONS="session_53" ./03_process_sessions.sh   # single session
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

echo "======================================================================"
echo "STEP 3/4: align + cut audio/video for every session (process_session.py)"
echo "======================================================================"

FORCE="${FORCE:-0}"          # set FORCE=1 to reprocess sessions that already have output
PARTS="${PARTS:-train dev}"  # override to restrict to e.g. "dev"
SESSIONS="${SESSIONS:-}"     # space-separated session names to restrict to (all if empty)

process_part() {
    local part="$1"
    local part_dir="$SRC_AVCOCKTAIL/$part"
    if [ ! -d "$part_dir" ]; then
        echo "[3/4] WARNING: $part_dir does not exist, skipping '$part' split."
        return
    fi

    # pick which sessions to iterate: either the caller's explicit list, or
    # every session_* directory found under this split.
    local sessions
    if [ -n "$SESSIONS" ]; then
        sessions="$SESSIONS"
    else
        sessions=$(find "$part_dir" -maxdepth 1 -mindepth 1 -type d -name "session_*" -exec basename {} \; | sort)
    fi

    local total
    total=$(echo "$sessions" | grep -c . || true)
    echo "[3/4] $part: $total session(s) to consider"

    local i=0
    for session in $sessions; do
        i=$((i + 1))
        local session_dir="$part_dir/$session"

        if [ ! -d "$session_dir" ]; then
            echo "  [$i/$total] $part/$session: no such session directory, skipping"
            continue
        fi

        # a session with no speaker_to_cluster.json has nothing to pair up
        if [ ! -f "$session_dir/labels/speaker_to_cluster.json" ]; then
            echo "  [$i/$total] $part/$session: missing labels/speaker_to_cluster.json, skipping"
            continue
        fi

        # idempotency: skip sessions that already have stereo output, unless
        # FORCE=1 (ffmpeg re-encoding a whole session is expensive).
        local stereo_dir="$session_dir/central-stereo_audios"
        if [ "$FORCE" != "1" ] && [ -d "$stereo_dir" ] && [ -n "$(ls -A "$stereo_dir" 2>/dev/null)" ]; then
            echo "  [$i/$total] $part/$session: central-stereo_audios already populated, skipping (FORCE=1 to redo)"
            continue
        fi

        echo "  [$i/$total] $part/$session: processing..."
        python3 "$SCRIPT_DIR/process_session.py" --root "$SRC_AVCOCKTAIL" --part "$part" --session "$session"
    done
}

for part in $PARTS; do
    process_part "$part"
done

echo "[3/4] done."

#!/usr/bin/env bash
# ==============================================================================
# 02_download_event_labels.sh -- STEP 2 of 4
# ------------------------------------------------------------------------------
# Downloads turn-taking event labels, channel-mapping pickles, and OpenFace
# feature pickles from:
#   https://huggingface.co/datasets/lggvu/avcocktail-turn-taking
#   https://huggingface.co/datasets/lggvu/avcocktail-openface-features
# and merges them into $SRC_AVCOCKTAIL, matching the existing session layout:
#   dev/<session>/annotated/events-<spkA>-<spkB>.json          (manually annotated)
#   dev/<session>/speakers/<spk>/annotated-pkl_fps30/<spk>.pkl (OpenFace features)
#   dev/central-dev_channelmaps_filtered.pkl
#   train/<session>/events-<spkA>-<spkB>.json                  (rule-based, unannotated)
#   train/<session>/speakers/<spk>/central-pkl_fps30/<spk>.pkl (OpenFace features)
#   train/central-train_channelmaps.pkl
#
# IMPORTANT: the .pkl files in that repo are stored via git-lfs. A plain
# `git clone` without the git-lfs binary installed silently checks out tiny
# LFS *pointer* text files instead of the real content (this bit us once --
# see the cleanup at the bottom). To avoid depending on git-lfs being
# installed at all, this script instead uses the `huggingface_hub` python
# package, which resolves LFS blobs over plain HTTPS.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

echo "======================================================================"
echo "STEP 2/4: download event label + OpenFace feature files"
echo "======================================================================"

EVENT_LABELS_REPO="lggvu/avcocktail-turn-taking"
OPENFACE_FEATURES_REPO="lggvu/avcocktail-openface-features"

if [ ! -d "$SRC_AVCOCKTAIL/train" ] || [ ! -d "$SRC_AVCOCKTAIL/dev" ]; then
    echo "[2/4] ERROR: $SRC_AVCOCKTAIL/train or /dev doesn't exist yet -- run step 1 first." >&2
    exit 1
fi

# ---- repair: remove any stray git-lfs pointer stubs from a previous plain
# `git clone` run (tiny ASCII files instead of the real binary content) so
# they get re-fetched for real below, instead of being mistaken for already-
# downloaded data.
echo "[2/4] checking for stray git-lfs pointer stubs from earlier runs..."
mapfile -t stub_files < <(grep -rlI "git-lfs.github.com/spec/v1" "$SRC_AVCOCKTAIL/train" "$SRC_AVCOCKTAIL/dev" 2>/dev/null || true)
if [ "${#stub_files[@]}" -gt 0 ]; then
    echo "[2/4]   found ${#stub_files[@]} pointer stub(s), removing so they're re-downloaded for real:"
    for f in "${stub_files[@]}"; do
        echo "[2/4]     rm $f"
        rm -f "$f"
    done
fi

# ---- fetch (huggingface_hub keeps its own manifest under $SRC_AVCOCKTAIL, so
# re-running this only downloads what's new/changed -- no separate "already
# downloaded" check needed here).
TOKEN_ARG=""
if [ -n "${HF_TOKEN:-}" ] && [ "$HF_TOKEN" != "your_actual_token_here" ]; then
    TOKEN_ARG="$HF_TOKEN"
fi

echo "[2/4] syncing $EVENT_LABELS_REPO -> $SRC_AVCOCKTAIL..."
conda run --no-capture-output -n "$CHUNKS_CONDA_ENV" python3 - "$EVENT_LABELS_REPO" "$SRC_AVCOCKTAIL" "$TOKEN_ARG" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir, token = sys.argv[1], sys.argv[2], sys.argv[3] or None
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    token=token,
    allow_patterns=["train/*", "dev/*"],
)
print("snapshot_download complete")
PY

echo "[2/4] syncing $OPENFACE_FEATURES_REPO -> $SRC_AVCOCKTAIL (this can take a while the first time, ~1GB)..."
conda run --no-capture-output -n "$CHUNKS_CONDA_ENV" python3 - "$OPENFACE_FEATURES_REPO" "$SRC_AVCOCKTAIL" "$TOKEN_ARG" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir, token = sys.argv[1], sys.argv[2], sys.argv[3] or None
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    token=token,
    allow_patterns=["train/*", "dev/*"],
)
print("snapshot_download complete")
PY

echo "[2/4] verifying no LFS pointer stubs remain..."
if grep -rlI "git-lfs.github.com/spec/v1" "$SRC_AVCOCKTAIL/train" "$SRC_AVCOCKTAIL/dev" 2>/dev/null; then
    echo "[2/4] ERROR: the above files are still LFS pointer stubs, not real content." >&2
    exit 1
fi

n_dev_events=$(find "$SRC_AVCOCKTAIL/dev" -path "*/annotated/events-*.json" 2>/dev/null | wc -l)
n_train_events=$(find "$SRC_AVCOCKTAIL/train" -maxdepth 2 -name "events-*.json" 2>/dev/null | wc -l)
n_train_pkl=$(find "$SRC_AVCOCKTAIL/train" -path "*/central-pkl_fps30/*.pkl" 2>/dev/null | wc -l)
n_dev_pkl=$(find "$SRC_AVCOCKTAIL/dev" -path "*/annotated-pkl_fps30/*.pkl" 2>/dev/null | wc -l)
echo "[2/4] done."
echo "[2/4]   dev:   $n_dev_events annotated event file(s), $n_dev_pkl OpenFace pkl(s)"
echo "[2/4]   train: $n_train_events rule-based event file(s), $n_train_pkl OpenFace pkl(s)"

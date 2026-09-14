#!/usr/bin/env bash
# ==============================================================================
# 01_download_dataset.sh -- STEP 1 of 4
# ------------------------------------------------------------------------------
# Downloads the AVCocktail/MCoRec dataset (train + dev splits, "without
# central videos" variants) from Hugging Face into $SRC_AVCOCKTAIL, unless
# it is already there.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

echo "======================================================================"
echo "STEP 1/4: download AVCocktail (MCoRec) dataset"
echo "======================================================================"

# ---- has this already been done? --------------------------------------------
# "already downloaded" = the folder exists, is non-empty, AND is bigger than
# 10MB. The size check guards against a stray empty dir or a failed/partial
# download being mistaken for a real dataset.
dir_is_populated() {
    local dir="$1"
    [ -d "$dir" ] || return 1
    [ -n "$(ls -A "$dir" 2>/dev/null)" ] || return 1
    local size_kb
    size_kb=$(du -sk "$dir" 2>/dev/null | cut -f1)
    [ "${size_kb:-0}" -gt 10240 ]  # 10MB in KB
}

if dir_is_populated "$SRC_AVCOCKTAIL"; then
    echo "[1/4] SRC_AVCOCKTAIL ($SRC_AVCOCKTAIL) already exists and is > 10MB."
    echo "[1/4] Skipping download. Remove/empty it first if you want to re-fetch."
    exit 0
fi

echo "[1/4] $SRC_AVCOCKTAIL is missing/empty/too small -> downloading from Hugging Face."

if [ -z "${HF_TOKEN:-}" ] || [ "$HF_TOKEN" = "your_actual_token_here" ]; then
    echo "[1/4] ERROR: set HF_TOKEN in 00_config.sh to your real Hugging Face token first." >&2
    echo "[1/4]        (get one at https://huggingface.co/settings/tokens and make sure" >&2
    echo "[1/4]         your account has access to MCoRecChallenge/MCoRec)" >&2
    exit 1
fi

mkdir -p "$SRC_AVCOCKTAIL"

DOWNLOAD_DIR="$(mktemp -d)"
cleanup() { echo "[1/4] cleaning up scratch dir $DOWNLOAD_DIR"; rm -rf "$DOWNLOAD_DIR"; }
trap cleanup EXIT
echo "[1/4] scratch download dir: $DOWNLOAD_DIR"

echo "[1/4] (1/4) downloading dev_without_central_videos.zip ..."
wget --header="Authorization: Bearer $HF_TOKEN" -O "$DOWNLOAD_DIR/dev_without_central_videos.zip" \
    "https://huggingface.co/datasets/MCoRecChallenge/MCoRec/resolve/main/dev_without_central_videos.zip"

echo "[1/4] (2/4) downloading train_without_central_videos.zip ..."
wget --header="Authorization: Bearer $HF_TOKEN" -O "$DOWNLOAD_DIR/train_without_central_videos.zip" \
    "https://huggingface.co/datasets/MCoRecChallenge/MCoRec/resolve/main/train_without_central_videos.zip"

echo "[1/4] (3/4) unzipping train split -> $SRC_AVCOCKTAIL/train ..."
mkdir -p "$SRC_AVCOCKTAIL/train"
unzip -q "$DOWNLOAD_DIR/train_without_central_videos.zip" -d "$SRC_AVCOCKTAIL/train"

echo "[1/4] (4/4) unzipping dev split -> $SRC_AVCOCKTAIL/dev ..."
mkdir -p "$SRC_AVCOCKTAIL/dev"
unzip -q "$DOWNLOAD_DIR/dev_without_central_videos.zip" -d "$SRC_AVCOCKTAIL/dev"

n_train=$(ls "$SRC_AVCOCKTAIL/train" 2>/dev/null | grep -c '^session_' || true)
n_dev=$(ls "$SRC_AVCOCKTAIL/dev" 2>/dev/null | grep -c '^session_' || true)
echo "[1/4] done. train sessions: $n_train, dev sessions: $n_dev"

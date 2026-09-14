#!/usr/bin/env bash
# ==============================================================================
# run_all.sh
# ------------------------------------------------------------------------------
# Runs the full AVCocktail data-prep pipeline, step by step:
#   1. 01_download_dataset.sh    download train+dev from Hugging Face (if needed)
#   2. 02_download_event_labels.sh   [currently a stub, see TODO inside it]
#   3. 03_process_sessions.sh    align+cut audio/video, build stereo wavs
#   4. 04_generate_chunks.sh     slice into model-ready chunks + VAD/VAP labels
#
# Edit 00_config.sh first (at least HF_TOKEN and SRC_AVCOCKTAIL), then run:
#   ./run_all.sh
#
# Each step script can also be run on its own; see the comments at the top of
# each file for step-specific options (FORCE, PARTS, SESSIONS, ...).
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "######################################################################"
echo "# AVCocktail data-prep pipeline"
echo "######################################################################"

"$SCRIPT_DIR/01_download_dataset.sh"
echo
"$SCRIPT_DIR/02_download_event_labels.sh"
echo
"$SCRIPT_DIR/03_process_sessions.sh"
echo
"$SCRIPT_DIR/04_generate_chunks.sh"

echo
echo "######################################################################"
echo "# pipeline complete"
echo "######################################################################"

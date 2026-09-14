#!/usr/bin/env bash
# ==============================================================================
# 00_config.sh
# ------------------------------------------------------------------------------
# Single source of truth for every constant the avcocktail_prep pipeline uses.
# Every other script in this folder starts with:
#     source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"
# so edit values HERE, not in the individual step scripts.
#
# Usage: edit the values below (at least HF_TOKEN), then run ./run_all.sh,
# or source this file yourself and run the numbered step scripts one by one.
# ==============================================================================

# ---- Hugging Face access ----------------------------------------------------
# Your personal Hugging Face access token (https://huggingface.co/settings/tokens).
# Needed to download the gated MCoRec/AVCocktail dataset in step 1. Your HF
# account must also have been granted access to the dataset itself:
# https://huggingface.co/datasets/MCoRecChallenge/MCoRec
export HF_TOKEN="your_actual_token_here"

# ---- dataset location --------------------------------------------------------
# Local root of the AVCocktail/MCoRec dataset. After step 1 it contains a
# train/ and dev/ folder, each with one subfolder per session (session_00, ...).
export SRC_AVCOCKTAIL="/home/lhoang/Programming/datasets/mcorec"

# ---- model input window ------------------------------------------------------
# Length (seconds) of the sliding audio window fed to the model, and the
# stride (seconds) between the start of consecutive windows. Step 4
# (generate_chunks_avcocktail.py) reads these two values from THIS file via
# environment variables, so changing them here is all that's needed to
# regenerate chunks with a different window.
export WINDOW_SIZE=20
export STEP_SIZE=5

# Extra knobs generate_chunks_avcocktail.py needs that weren't listed as
# top-level constants for this task; override if needed.
export FUTURE_CONTEXT=2      # seconds of future VAD kept per window (for the VAP head)
export AUDIO_FEATURE_HZ=50   # frame rate the VAD one-hot labels are stored at

# ---- step 4 output location --------------------------------------------------
# Where the generated manifest CSVs + chunk tensors (.npy / .pt) get written.
export CHUNKS_OUT_DIR="${SRC_AVCOCKTAIL}/_chunks"

# ---- step 4 python environment -----------------------------------------------
# Conda env with the packages generate_chunks_avcocktail.py needs (pydub,
# textgrid, einops, pandas, torch). NOTE (see project memory): the `mmvap`
# env has a broken torch install -- use `turn_taking` instead.
export CHUNKS_CONDA_ENV="turn_taking"

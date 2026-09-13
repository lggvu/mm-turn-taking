"""Run OpenFace (via the algebr/openface:latest Docker image) on a
single-speaker video and convert its output CSV into the 60-dim per-frame
feature vector this project's video-capable VAP models were trained on.

Column set/order and per-file normalization verified directly against real
training pkls: gaze(6) + pose(6) + landmarks(30: jaw/brow/nose) are min-max
normalized then mean-centered per file; AU intensities(17) + confidence are
kept raw.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import torch

GAZE_COLS = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z"]
POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
JAW_COLS = ["x_3", "x_13", "x_6", "x_10", "x_8", "y_3", "y_13", "y_6", "y_10", "y_8"]
BROW_COLS = ["x_19", "x_24", "y_19", "y_24"]
NOSE_COLS = [f"x_{i}" for i in range(27, 35)] + [f"y_{i}" for i in range(27, 35)]
LMK_COLS = JAW_COLS + BROW_COLS + NOSE_COLS  # 30
FAU_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r",
    "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]  # 17
NORMALIZE_COLS = GAZE_COLS + POSE_COLS + LMK_COLS  # 42
ALL_COLUMNS = GAZE_COLS + POSE_COLS + LMK_COLS + FAU_COLS + ["confidence"]  # 60 == 'all' feature_set

OPENFACE_IMAGE = "algebr/openface:latest"
VIDEO_FPS = 30.0  # matches the fixed fps assumed throughout mmvap/data/dataloader.py


def _reencode_for_openface(video_path: str, out_path: str, fps: float = VIDEO_FPS) -> str:
    """OpenFace's docker image (older OpenCV/ffmpeg build) has been observed
    to silently truncate tracking to 1-2 frames on some modern H.264
    encodings before segfaulting during its own cleanup step. Re-encoding to
    a plain baseline-profile yuv420p mp4 at a fixed fps first fixes that,
    and also guarantees both speakers' features end up on the same clock."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-an",
            str(out_path),
        ],
        check=True, capture_output=True,
    )
    return str(out_path)


def run_openface_docker(video_path: str, out_dir: str) -> str:
    """Run OpenFace FeatureExtraction (single face per video, as this
    project's data always has) and return the path to the resulting CSV."""
    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        _reencode_for_openface(video_path, Path(tmp_dir) / "input.mp4")

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{tmp_dir}:/data:ro",
                "-v", f"{out_dir}:/out",
                "--entrypoint", "/bin/sh",
                OPENFACE_IMAGE,
                "-c", 'build/bin/FeatureExtraction -f "/data/input.mp4" -out_dir /out',
            ],
            capture_output=True, text=True,
        )
        csv_path = out_dir / "input.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"OpenFace did not produce a CSV for '{video_path}' (is Docker running, and is "
                f"'{OPENFACE_IMAGE}' pulled?).\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        # OpenFace's own AU-postprocessing cleanup occasionally segfaults
        # *after* the CSV is already written -- a nonzero exit code here
        # doesn't mean the CSV is bad, so we only fail on it missing above.

        # The container runs as root, so everything it wrote under out_dir
        # is root-owned on the host; hand it back to the calling user.
        subprocess.run(
            [
                "docker", "run", "--rm", "-v", f"{out_dir}:/out",
                "--entrypoint", "/bin/sh", OPENFACE_IMAGE,
                "-c", f"chown -R {os.getuid()}:{os.getgid()} /out",
            ],
            capture_output=True,
        )
        return str(csv_path)


def csv_to_feature_frame(csv_path: str) -> pd.DataFrame:
    """Convert a raw OpenFace CSV into the 60-column, per-file-normalized
    feature frame this project's video-capable models expect."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.astype(float)
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OpenFace CSV '{csv_path}' is missing expected columns: {missing}")
    df = df[ALL_COLUMNS].copy()

    norm = df[NORMALIZE_COLS]
    span = (norm.max() - norm.min()).replace(0, 1.0)
    norm = (norm - norm.min()) / span
    norm = norm - norm.mean()
    df[NORMALIZE_COLS] = norm
    return df


def get_video_features(video_path: str, out_dir: str, label: str = "speaker", csv_path: str = None):
    """End to end: run OpenFace (unless csv_path is already supplied),
    convert to the 60-dim feature frame, save it as a pkl (so it's reusable
    with the existing dataloader classes / inspectable for debugging), and
    return (features [N, 60] float32 tensor, pkl_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is None:
        csv_path = run_openface_docker(video_path, out_dir / f"openface_{label}")

    df = csv_to_feature_frame(csv_path)
    pkl_path = out_dir / f"{label}_features.pkl"
    df.to_pickle(pkl_path)

    features = torch.tensor(df.values, dtype=torch.float32)
    return features, str(pkl_path)

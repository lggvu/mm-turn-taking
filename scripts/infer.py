"""Off-the-shelf VAP inference on a single audio/video file.

Given just a checkpoint and an input file, this figures out which model
class the checkpoint holds, runs it end to end, and (optionally) renders a
synced visualization video. No config file, channelmap, or pre-extracted
features required -- video features are extracted with OpenFace on the fly.

Audio-only models (StereoTransformerModel) need one stereo file (or a video
whose audio track is already stereo, e.g. an ego-video with a Left/Right mic
per speaker):

    python scripts/infer.py --checkpoint runs/vap/last.ckpt --input call.wav --visualize

Video-capable models (StereoTransformerModelVideoOnly, EarlyVAFusion,
LateVAFusion) need one video per speaker, since the architecture makes a
per-speaker prediction from each speaker's own face:

    python scripts/infer.py --checkpoint runs/mmvap/last.ckpt \\
        --input speakerA.mp4 --video-right speakerB.mp4 --visualize

If the two speakers' videos already carry their own audio, the stereo audio
track is built for you automatically (each video's audio -> one channel).
Pass --audio explicitly to override that.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from inference.audio_utils import build_stereo_audio_from_videos, load_stereo_audio
from inference.model_loader import load_model
from inference.openface_utils import get_video_features
from inference.pipeline import WindowedVAPInput, decode_predictions, run_streaming_inference
from inference.visualize import render_visualization


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.ckpt).")
    p.add_argument("--input", required=True,
                   help="Path to the audio/video input. For video-capable models this is "
                        "speaker A's video; pass speaker B's with --video-right.")
    p.add_argument("--video-right", default=None,
                   help="Speaker B's video (required for video-capable models).")
    p.add_argument("--audio", default=None,
                   help="Explicit stereo audio path, overriding whatever would otherwise be "
                        "read from --input / derived from the two videos.")
    p.add_argument("--openface-csv-left", default=None, help="Reuse an existing OpenFace CSV instead of running Docker for speaker A.")
    p.add_argument("--openface-csv-right", default=None, help="Reuse an existing OpenFace CSV instead of running Docker for speaker B.")
    p.add_argument("--model-class", default=None,
                   help="Override automatic model-class detection (one of the classes in "
                        "inference.model_loader.MODEL_CLASSES).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None, help="Default: outputs/<input-stem>_<checkpoint-stem>/")
    p.add_argument("--window-size", type=float, default=20.0,
                   help="Model sliding-window size in seconds; also how much time the "
                        "--visualize view shows at once (it streams/scrolls rather than "
                        "showing the whole file).")
    p.add_argument("--step-size", type=float, default=19.0, help="Sliding-window step in seconds.")
    p.add_argument("--feature-sr", type=int, default=50, help="Model output frame rate (Hz).")
    p.add_argument("--visualize", action="store_true", help="Render a synced waveform/video + VAD/VAP probability video.")
    p.add_argument("--viz-fps", type=float, default=10.0, help="Frame rate of the rendered visualization video.")
    p.add_argument("--viz-max-seconds", type=float, default=None, help="Cap the visualization to the first N seconds (inference itself always covers the whole input).")
    return p


def main():
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"--input not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "outputs" / f"{input_path.stem}_{Path(args.checkpoint).stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer] output dir: {output_dir}")

    model, model_name, mode, modality = load_model(args.checkpoint, device=args.device, model_class=args.model_class)

    video_l_feat = video_r_feat = None
    video_left_path = video_right_path = None
    if modality["video"]:
        if args.video_right is None and args.openface_csv_right is None:
            raise ValueError(
                f"'{model_name}' needs a video per speaker. Pass --video-right for speaker B "
                f"(--input is used as speaker A's video)."
            )
        video_left_path = str(input_path)
        video_right_path = args.video_right

        print("[infer] extracting speaker A's video features (OpenFace)...")
        video_l_feat, _ = get_video_features(video_left_path, output_dir, label="left", csv_path=args.openface_csv_left)
        print(f"[infer]   {video_l_feat.shape[0]} frames")

        print("[infer] extracting speaker B's video features (OpenFace)...")
        video_r_feat, _ = get_video_features(video_right_path, output_dir, label="right", csv_path=args.openface_csv_right)
        print(f"[infer]   {video_r_feat.shape[0]} frames")

    audio = None
    audio_playback_path = None
    if modality["audio"]:
        if args.audio:
            audio, audio_playback_path = load_stereo_audio(args.audio, work_dir=output_dir)
        elif modality["video"]:
            print("[infer] no --audio given; building stereo audio from the two videos' own audio tracks...")
            stereo_wav = build_stereo_audio_from_videos(video_left_path, video_right_path, output_dir)
            audio, audio_playback_path = load_stereo_audio(stereo_wav, work_dir=output_dir)
        else:
            audio, audio_playback_path = load_stereo_audio(str(input_path), work_dir=output_dir)
    elif args.visualize:
        # Model doesn't need audio, but grab some for the visualization if we can.
        try:
            source = args.audio or (
                build_stereo_audio_from_videos(video_left_path, video_right_path, output_dir)
                if modality["video"] else str(input_path)
            )
            _, audio_playback_path = load_stereo_audio(source, work_dir=output_dir)
        except Exception as e:
            print(f"[infer] no usable audio for the visualization track ({e}); it will be silent.")

    dataset = WindowedVAPInput(
        audio=audio, video_l=video_l_feat, video_r=video_r_feat,
        sr=16000, video_fps=30.0, window_size=args.window_size, step_size=args.step_size,
    )
    print(f"[infer] running inference over {dataset.duration:.1f}s in {len(dataset)} window(s)...")
    vaps, vads = run_streaming_inference(model, dataset, mode=mode, feature_sr=args.feature_sr)
    predictions = decode_predictions(vaps, mode=mode)
    p_future, p_now = predictions["p_future"], predictions["p_now"]
    print(f"[infer] produced {p_future.shape[0]} prediction frames ({p_future.shape[0] / args.feature_sr:.1f}s)")

    np.savez(
        output_dir / "predictions.npz",
        p_future=p_future.numpy(), p_now=p_now.numpy(), p_bc=predictions["p_bc"].numpy(),
        vad=vads.numpy(), feature_sr=args.feature_sr,
    )
    summary = {
        "checkpoint": str(args.checkpoint), "model_name": model_name, "mode": mode,
        "modality": modality, "duration_sec": dataset.duration,
        "num_prediction_frames": int(p_future.shape[0]), "feature_sr": args.feature_sr,
        "window_size": args.window_size, "step_size": args.step_size,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[infer] saved predictions.npz and summary.json to {output_dir}")

    if args.visualize:
        viz_path = output_dir / "visualization.mp4"
        print(f"[infer] rendering visualization to {viz_path} ...")
        render_visualization(
            output_path=viz_path,
            p_future=p_future.numpy(),
            vad=vads.numpy(),
            feature_sr=args.feature_sr,
            audio_path=audio_playback_path,
            video_left_path=video_left_path,
            video_right_path=video_right_path,
            fps=args.viz_fps,
            max_seconds=args.viz_max_seconds,
            window_size=args.window_size,
        )
        print(f"[infer] wrote {viz_path}")


if __name__ == "__main__":
    import time
    s_time = time.time()
    main()
    e_time = time.time()
    print(f"[infer] total time: {e_time - s_time:.1f}s")

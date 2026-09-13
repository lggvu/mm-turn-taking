"""Render a synced "real-time" visualization video: waveform(s) and/or the
source video(s), plus a streaming VAD/VAP probability view that only ever
shows the current `window_size`-second window (like a live monitor), muxed
with the original audio.

Adapted from the working pattern in mmvap-lightning/demo/infer_sample.py's
render_visualization, generalized to 0/1/2 video panels and to always draw
the audio waveform panel (not just the video), and simplified to always
cover the whole input rather than an arbitrary crop.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _load_waveform_for_plot(audio_path: str, target_points: int = 4000):
    import soundfile as sf

    data, sr = sf.read(audio_path, always_2d=True)  # [N, C]
    n = data.shape[0]
    step = max(1, n // target_points)
    data = data[::step]
    times = np.arange(data.shape[0]) * step / sr
    return times, data, n / sr


def _capture_frame(cv2_module, cap, target_size=None):
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
    if target_size is not None:
        frame = cv2_module.resize(frame, target_size, interpolation=cv2_module.INTER_AREA)
    return frame


def render_visualization(
    output_path,
    p_future: np.ndarray,
    vad: np.ndarray,
    feature_sr: int,
    audio_path: str = None,
    video_left_path: str = None,
    video_right_path: str = None,
    left_label: str = "Speaker 0",
    right_label: str = "Speaker 1",
    fps: float = 10.0,
    max_seconds: float = None,
    window_size: float = 10.0,
):
    """Write an mp4 to `output_path`. Panels included depend on what's
    available: video panel(s) if video_left_path/video_right_path are given,
    a waveform panel if audio_path is given, and always VAD + VAP traces.

    The waveform/probability panels stream: at any instant only the most
    recent `window_size` seconds are visible, scrolling forward as playback
    proceeds (rather than showing the whole file with a moving playhead)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = p_future.shape[0] / feature_sr
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    n_pred_frames = int(duration * feature_sr)
    times = np.arange(n_pred_frames) / feature_sr
    p_left, p_right = p_future[:n_pred_frames, 0], p_future[:n_pred_frames, 1]
    vad_left, vad_right = vad[:n_pred_frames, 0], vad[:n_pred_frames, 1]

    view_span = min(window_size, duration) if duration > 0 else window_size

    has_audio = audio_path is not None

    # Probe the videos (and grab their first frame) *before* laying out the
    # figure, since a video that opens but that cv2 can't actually decode
    # (corrupt/unusual container -- has happened with concat-copied mp4s)
    # should just fall back to no video panel rather than crashing.
    left_cap = right_cap = None
    video_fps = None
    first_l = first_r = None
    if video_left_path is not None and video_right_path is not None:
        import cv2

        left_cap = cv2.VideoCapture(str(video_left_path))
        right_cap = cv2.VideoCapture(str(video_right_path))
        video_fps = float(min(left_cap.get(cv2.CAP_PROP_FPS) or 25.0, right_cap.get(cv2.CAP_PROP_FPS) or 25.0))
        first_l = _capture_frame(cv2, left_cap)
        first_r = _capture_frame(cv2, right_cap)
        if first_l is None or first_r is None:
            print(f"[visualize] could not decode frames from the source video(s); "
                  f"skipping the video panel.")
            left_cap.release()
            right_cap.release()
            left_cap = right_cap = None

    has_video = left_cap is not None

    n_rows = 1 + (1 if has_video else 0) + (1 if has_audio else 0)
    height_ratios = ([3] if has_video else []) + ([1] if has_audio else []) + [2]
    fig = plt.figure(figsize=(16, 3.2 * n_rows + 1))
    grid = fig.add_gridspec(n_rows, 2, height_ratios=height_ratios, hspace=0.4, wspace=0.15)

    row = 0
    left_image = right_image = None
    if has_video:
        import cv2

        ax_l = fig.add_subplot(grid[row, 0])
        ax_r = fig.add_subplot(grid[row, 1])
        ax_l.axis("off")
        ax_r.axis("off")
        h = max(first_l.shape[0], first_r.shape[0])
        l_size = (int(first_l.shape[1] * h / first_l.shape[0]), h)
        r_size = (int(first_r.shape[1] * h / first_r.shape[0]), h)
        first_l = cv2.resize(first_l, l_size)
        first_r = cv2.resize(first_r, r_size)
        left_image = ax_l.imshow(first_l)
        right_image = ax_r.imshow(first_r)
        ax_l.set_title(left_label, fontsize=12, fontweight="bold", color="#e74c3c")
        ax_r.set_title(right_label, fontsize=12, fontweight="bold", color="#3498db")
        row += 1

    wave_cursors = []
    if has_audio:
        wtimes, wdata, wdur = _load_waveform_for_plot(audio_path)
        mask = wtimes <= duration
        ax_wave = fig.add_subplot(grid[row, :])
        ax_wave.plot(wtimes[mask], wdata[mask, 0], color="#e74c3c", linewidth=0.6, alpha=0.8, label=left_label)
        if wdata.shape[1] > 1:
            ax_wave.plot(wtimes[mask], -wdata[mask, 1], color="#3498db", linewidth=0.6, alpha=0.8, label=right_label)
        ax_wave.set_xlim(0, view_span)
        ax_wave.set_ylabel("Waveform")
        ax_wave.set_yticks([])
        ax_wave.legend(loc="upper right", fontsize=8, ncol=2)
        wave_cursors.append(ax_wave.axvline(0, color="black", linestyle="--", linewidth=1.2))
        row += 1

    ax_prob = fig.add_subplot(grid[row, :])
    ax_prob.plot(times, vad_left, color="#e74c3c", linewidth=1.2, alpha=0.5, linestyle=":", label=f"VAD {left_label}")
    ax_prob.plot(times, vad_right, color="#3498db", linewidth=1.2, alpha=0.5, linestyle=":", label=f"VAD {right_label}")
    ax_prob.plot(times, p_left, color="#e74c3c", linewidth=2.0, label=f"VAP {left_label}")
    ax_prob.plot(times, p_right, color="#3498db", linewidth=2.0, label=f"VAP {right_label}")
    ax_prob.set_xlim(0, view_span)
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_xlabel("Time (s)")
    ax_prob.set_ylabel("Probability")
    ax_prob.grid(True, alpha=0.25)
    ax_prob.legend(loc="upper right", fontsize=8, ncol=4)
    ax_prob.set_title("VAD (dotted) / next-speaker VAP (solid)", fontsize=11, fontweight="bold")
    prob_cursor = ax_prob.axvline(0, color="black", linestyle="--", linewidth=1.2)

    playhead_text = fig.text(0.5, 0.98, "t = 0.00s", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    n_out_frames = max(1, int(duration * fps))

    def update(i):
        t = min(i / fps, duration)
        view_hi = max(t, view_span)
        view_lo = view_hi - view_span
        if wave_cursors:
            wave_cursors[0].axes.set_xlim(view_lo, view_hi)
        ax_prob.set_xlim(view_lo, view_hi)
        if has_video:
            target_src_frame = int(round(t * video_fps))
            cur_l = int(left_cap.get(1))  # CAP_PROP_POS_FRAMES
            cur_r = int(right_cap.get(1))
            import cv2
            if target_src_frame > cur_l:
                left_cap.set(cv2.CAP_PROP_POS_FRAMES, target_src_frame)
            if target_src_frame > cur_r:
                right_cap.set(cv2.CAP_PROP_POS_FRAMES, target_src_frame)
            fl = _capture_frame(cv2, left_cap, l_size)
            fr = _capture_frame(cv2, right_cap, r_size)
            if fl is not None:
                left_image.set_data(fl)
            if fr is not None:
                right_image.set_data(fr)
        for c in wave_cursors:
            c.set_xdata([t, t])
        prob_cursor.set_xdata([t, t])
        playhead_text.set_text(f"t = {t:.2f}s")
        artists = [prob_cursor, playhead_text] + wave_cursors
        if has_video:
            artists += [left_image, right_image]
        return artists

    with tempfile.TemporaryDirectory() as tmp_dir:
        silent_path = Path(tmp_dir) / "silent.mp4"
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"])
        anim = FuncAnimation(fig, update, frames=n_out_frames, blit=False)
        anim.save(str(silent_path), writer=writer, dpi=120)
        plt.close(fig)
        if has_video:
            left_cap.release()
            right_cap.release()

        if has_audio:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-t", f"{duration:.6f}", "-i", str(audio_path),
                    "-i", str(silent_path),
                    "-map", "1:v:0", "-map", "0:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart",
                    str(output_path),
                ],
                check=True, capture_output=True,
            )
        else:
            silent_path.replace(output_path)

    return output_path

"""Audio loading/extraction helpers for single-file VAP inference."""
import subprocess
from pathlib import Path

import torch

from mmvap.data.audio_manager import AudioManager


def load_stereo_audio(path: str, sr: int = 16000, work_dir: str = None):
    """Load a stereo audio track as [N, 2] (channel 0 = speaker A, channel 1
    = speaker B). `path` may be a wav or any container ffmpeg can demux
    (mp4, mov, ...) -- it's always passed through ffmpeg first so torchaudio
    (which only reads plain audio files) never has to touch a video
    container directly, and so callers get back a plain wav path too (e.g.
    for visualization, which can't read video containers as audio either).

    Returns (audio [N, 2] tensor, extracted_wav_path).
    """
    work_dir = Path(work_dir) if work_dir else Path(path).resolve().parent
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / f"_extracted_{Path(path).stem}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav_path)],
        check=True, capture_output=True,
    )
    audio, _ = AudioManager.load_waveform(str(wav_path), sample_rate=sr, normalize=True)
    if audio.shape[1] == 1:
        raise ValueError(
            f"Audio '{path}' is mono, but VAP models need one channel per speaker "
            f"(stereo). Provide a 2-channel wav, or two per-speaker videos so their "
            f"audio tracks can be combined into a stereo track automatically."
        )
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, str(wav_path)


def extract_audio_track(video_path: str, out_wav: str, sr: int = 16000) -> str:
    """Extract a video's audio track as mono PCM wav via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        check=True, capture_output=True,
    )
    return str(out_wav)


def combine_mono_to_stereo(left_wav: str, right_wav: str, out_wav: str) -> str:
    """Merge two mono wavs into one stereo wav (left -> ch0, right -> ch1)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(left_wav), "-i", str(right_wav),
            "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]",
            "-map", "[a]", "-ac", "2", "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        check=True, capture_output=True,
    )
    return str(out_wav)


def build_stereo_audio_from_videos(video_left: str, video_right: str, out_dir: str, sr: int = 16000) -> str:
    """Convenience path used when the user gives two per-speaker videos and no
    explicit --audio: pull each video's own audio track and merge them into
    the stereo wav the model expects."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    left_wav = out_dir / "left_audio.wav"
    right_wav = out_dir / "right_audio.wav"
    extract_audio_track(video_left, left_wav, sr=sr)
    extract_audio_track(video_right, right_wav, sr=sr)
    stereo_wav = out_dir / "stereo_audio.wav"
    combine_mono_to_stereo(str(left_wav), str(right_wav), str(stereo_wav))
    return str(stereo_wav)

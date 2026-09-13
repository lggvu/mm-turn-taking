"""Windowed streaming inference for a single audio/video input.

This intentionally does not reuse mmvap.data.dataloader's Dataset classes:
those are built around a specific corpus layout (channelmaps, per-session
directories) that a one-off input file doesn't have. Here we just hold the
already-loaded audio/video tensors in memory and slice overlapping windows
directly, which is all a single file needs.
"""
import math
from typing import Optional

import torch

from mmvap.data.codebook import VAPDecoder


class WindowedVAPInput:
    """Slices audio (and/or per-speaker video features) into overlapping
    windows for streaming VAP inference, the same way
    ValidationAudioDataset/ValidationAudioVisualDataset do for corpus files,
    but directly from in-memory tensors.

    Args:
        audio: [N_samples, 2] or None.
        video_l, video_r: [N_frames, D] per-speaker video features, or None.
        sr: audio sample rate.
        video_fps: video feature frame rate.
        window_size, step_size: in seconds.
    """

    def __init__(
        self,
        audio: Optional[torch.Tensor] = None,
        video_l: Optional[torch.Tensor] = None,
        video_r: Optional[torch.Tensor] = None,
        sr: int = 16000,
        video_fps: float = 30.0,
        window_size: float = 20.0,
        step_size: float = 19.0,
    ):
        if audio is None and (video_l is None or video_r is None):
            raise ValueError("Need audio, or both video_l and video_r.")

        self.sr = sr
        self.video_fps = video_fps
        self.window_size = window_size
        self.step_size = step_size

        self.audio = audio
        self.video = None
        if video_l is not None and video_r is not None:
            n = min(video_l.shape[0], video_r.shape[0])
            self.video = torch.stack([video_l[:n], video_r[:n]], dim=-1)  # [N, D, 2]

        durations = []
        if self.audio is not None:
            durations.append(self.audio.shape[0] / sr)
        if self.video is not None:
            durations.append(self.video.shape[0] / video_fps)
        self.duration = min(durations)

        # Always slice/pad to *exactly* these full-window sizes (rather than
        # whatever a short/leftover window's true content happens to be).
        # The video branch upsamples with a scale_factor fixed at model-build
        # time from cfg's audio/video sequence_len ratio; only an exact
        # full-size window reproduces the input lengths (and their integer
        # up/downsampled lengths) the model was actually trained/validated
        # with, so this sidesteps float-rounding mismatches between the
        # audio encoder's downsampled length and the video branch's upsampled
        # length for any other window size. Any padded tail is trimmed back
        # off the output afterwards (see run_streaming_inference).
        self.audio_window_samples = int(round(self.window_size * sr))
        self.video_window_frames = int(round(self.window_size * video_fps))

        if self.duration <= self.window_size:
            self.n_windows = 1
        else:
            remaining = self.duration - self.window_size
            self.n_windows = 1 + int(math.ceil(remaining / self.step_size - 1e-9))

    def __len__(self):
        return self.n_windows

    @staticmethod
    def _slice_and_pad(x: torch.Tensor, start: int, length: int) -> torch.Tensor:
        chunk = x[start:start + length]
        if chunk.shape[0] < length:
            pad = chunk.new_zeros((length - chunk.shape[0], *chunk.shape[1:]))
            chunk = torch.cat([chunk, pad], dim=0)
        return chunk

    def __getitem__(self, idx):
        start = idx * self.step_size
        item = {"start": start, "end": min(start + self.window_size, self.duration)}

        if self.audio is not None:
            s = int(round(start * self.sr))
            item["audio_chunk"] = self._slice_and_pad(self.audio, s, self.audio_window_samples)
        if self.video is not None:
            s = int(round(start * self.video_fps))
            item["frames"] = self._slice_and_pad(self.video, s, self.video_window_frames)

        return item


def run_streaming_inference(model, dataset: WindowedVAPInput, mode: str = "VAP", feature_sr: int = 50):
    """Run the model over every window of `dataset`, dropping the
    already-seen overlap from every window after the first (standard
    sliding-window aggregation, matching eval/model_inference.py).

    Returns (vaps, vads) tensors of shape [T, ...] covering the whole input.
    """
    overlap = int(round(feature_sr * (dataset.window_size - dataset.step_size)))
    device = next(model.parameters()).device

    vaps_list, vads_list = [], []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        batch = {}
        if "audio_chunk" in sample:
            batch["audio_chunk"] = sample["audio_chunk"].unsqueeze(0).to(device)
        if "frames" in sample:
            batch["frames"] = sample["frames"].unsqueeze(0).to(device)

        with torch.no_grad():
            vad_pred, vap_pred = model(batch)

        vap_pred = (torch.softmax(vap_pred, dim=-1) if mode == "VAP" else torch.sigmoid(vap_pred)).cpu()
        vad_pred = torch.sigmoid(vad_pred).cpu()

        n = vap_pred.shape[1]
        this_overlap = min(overlap, n)
        if idx == 0:
            vaps_list.append(vap_pred[0, :this_overlap, ...])
            vads_list.append(vad_pred[0, :this_overlap, :])
        vaps_list.append(vap_pred[0, this_overlap:, ...])
        vads_list.append(vad_pred[0, this_overlap:, :])

    vaps = torch.cat(vaps_list, dim=0)
    vads = torch.cat(vads_list, dim=0)

    # The last window is padded up to a full window_size (see WindowedVAPInput);
    # drop whatever tail that padding produced beyond the input's real duration.
    valid_frames = int(round(dataset.duration * feature_sr))
    vaps, vads = vaps[:valid_frames], vads[:valid_frames]
    return vaps, vads


def decode_predictions(vaps: torch.Tensor, mode: str = "VAP") -> dict:
    """Decode raw per-frame VAP output into interpretable probabilities.
    Mirrors eval/model_inference.py::decode_predictions."""
    if vaps.ndim == 3:
        vaps = vaps.squeeze(0)

    if mode == "VAP":
        decoder = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])
        return {
            "p_future": decoder.p_future(vaps),
            "p_now": decoder.p_now(vaps),
            "p_bc": decoder.p_bc(vaps),
            "p_all": decoder.decode_probabilities(vaps),
        }

    if mode in ("ind-4", "ind-40"):
        num_bins = 4 if mode == "ind-4" else 40
        vaps_reshaped = vaps.reshape(vaps.shape[0], 2, num_bins)

        p_now_ch0 = vaps_reshaped[:, 0, 0:2].sum(dim=1)
        p_now_ch1 = vaps_reshaped[:, 1, 0:2].sum(dim=1)
        p_now = torch.stack([p_now_ch0, p_now_ch1], dim=1)

        p_future_ch0 = vaps_reshaped[:, 0, 2:4].sum(dim=1)
        p_future_ch1 = vaps_reshaped[:, 1, 2:4].sum(dim=1)
        p_future = torch.stack([p_future_ch0, p_future_ch1], dim=1)

        p_future = p_future / (p_future.sum(-1, keepdim=True) + 1e-5)
        p_now = p_now / (p_now.sum(-1, keepdim=True) + 1e-5)
        vaps_reshaped = vaps_reshaped / (vaps_reshaped.sum(-1, keepdim=True) + 1e-5)

        return {"p_future": p_future, "p_now": p_now, "p_bc": p_future, "p_all": vaps_reshaped}

    raise ValueError(f"Unknown mode: {mode}. Expected 'VAP', 'ind-4', or 'ind-40'")

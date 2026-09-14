"""
Self-contained TextGrid / VAD one-hot helpers.

Standalone re-implementation of the two functions
generate_chunks_avcocktail.py actually needs from mm-vap's
dataset_management/dataset_manager/scripts/transcript_processing.py, so this
public repo has no dependency on that private codebase.
"""

import collections

import textgrid
import torch
from einops import rearrange


def vads_from_transcript(transcript_file, samples, sample_rate=16000):
    """Parse a 2-speaker-tier TextGrid into voice-activity intervals.

    Returns [tier0_vads, tier1_vads], each a list of [start, end] pairs
    (seconds, or sample indices if samples=True).
    """
    tg = textgrid.TextGrid.fromFile(transcript_file)
    vad_list = collections.defaultdict(list)

    for i, tier in enumerate(tg.tiers):
        for interval in tier.intervals:
            if interval.mark == "":
                continue
            start, end = interval.minTime, interval.maxTime
            if samples:
                start, end = int(start * sample_rate), int(end * sample_rate)
            vad_list[i].append([start, end])

    return [vad_list[0], vad_list[1]]


def vad_list_to_one_hot(vad_list, track_size, samples, sr=None, start=0):
    """Convert a 2-channel list of [start, end] voice-activity intervals into
    a one-hot encoded (N_frames, 2) tensor."""
    output = torch.zeros(track_size)

    if len(vad_list) > 1:
        for i, channel_vad in enumerate(vad_list):
            for v in channel_vad:
                if samples:
                    output[i, v[0] - start:v[1] - start] = 1.0
                else:
                    output[i, int((v[0] - start) * sr):int((v[1] - start) * sr)] = 1.0

    return rearrange(output, "c n -> n c")

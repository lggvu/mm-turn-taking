"""
Generate sliding-window audio chunks + VAD/VAP labels for AVCocktail (MCoRec).
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from tqdm import tqdm

import transcript_utils

STEREO_SUBDIR = "central-stereo_audios"

# ---- constants read from THIS pipeline's bash scripts (00_config.sh) --------
# (falls back to sensible defaults if unset, so the script still works standalone)
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 20))
STEP_SIZE = int(os.environ.get("STEP_SIZE", 5))
FUTURE_CONTEXT = int(os.environ.get("FUTURE_CONTEXT", 2))
AUDIO_FEATURE_HZ = int(os.environ.get("AUDIO_FEATURE_HZ", 50))

TRACK_SIZE_WINDOW = [2, AUDIO_FEATURE_HZ * WINDOW_SIZE]
TRACK_SIZE_FUTURE_WINDOW = [2, AUDIO_FEATURE_HZ * FUTURE_CONTEXT]


def build_manifest(audio_dir, sessions=None):
    """Scan <audio_dir>/session_*/central-stereo_audios/*.wav and build the
    same "id" list the original script expected from a fold CSV: one row per
    stereo wav, formatted as "<session>**<wav_stem>" (e.g.
    "session_00**spk_3-spk_4-30fps_fixed")."""
    ids = []
    session_names = sorted(
        s for s in os.listdir(audio_dir)
        if s.startswith("session_") and os.path.isdir(os.path.join(audio_dir, s))
    )
    if sessions:
        wanted = set(sessions)
        session_names = [s for s in session_names if s in wanted]

    for session in session_names:
        stereo_dir = os.path.join(audio_dir, session, STEREO_SUBDIR)
        if not os.path.isdir(stereo_dir):
            continue
        for fname in sorted(os.listdir(stereo_dir)):
            if fname.endswith(".wav"):
                stem = fname[: -len(".wav")]
                ids.append(f"{session}**{stem}")
    return ids


def get_chunked_dataset(ids, audio_dir):
    """generate a sliding window dataset, with possibly overlapping windows
    (see WINDOW_SIZE / STEP_SIZE above)

    Args:
        ids (list[str]): "<session>**<wav_stem>" entries, e.g. from build_manifest()
        audio_dir (str): path to the split directory (contains session_* folders)

    Returns:
        ids, starts, ends, vads, vaps -- same shapes as the original script.
    """
    window_size = WINDOW_SIZE
    stride_size = STEP_SIZE

    out_ids = []
    starts = []
    ends = []
    vads = None
    vaps = None

    pbar = tqdm(total=len(ids))
    for id in ids:
        session, audio_name = id.split("**")

        # process_session.py (step 3) always writes stereo wavs under
        # central-stereo_audios/, for both train and dev.
        audio_file = os.path.join(audio_dir, session, STEREO_SUBDIR, f"{audio_name}.wav")

        audio = AudioSegment.from_file(audio_file)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000.0 - FUTURE_CONTEXT

        start_times_all = list(range(0, int(duration_seconds - stride_size), stride_size))
        end_times_all = [f + window_size for f in start_times_all]

        start_times = []
        end_times = []
        # remove the last incomplete window(s) so every window fully fits
        for start, end in zip(start_times_all, end_times_all):
            if end < duration_seconds:
                start_times.append(start)
                end_times.append(end)

        # the stereo wav is named "<spkA>-<spkB>-30fps[_fixed]"; the TextGrid
        # is just "<spkA>-<spkB>.TextGrid" -- strip both suffixes.
        tg_name = audio_name.replace("-30fps", "").replace("_fixed", "")
        transcript_file = os.path.join(audio_dir, session, f"{tg_name}.TextGrid")

        try:
            vad_list_master = transcript_utils.vads_from_transcript(transcript_file, samples=False)
        except Exception as e:
            print(f"missing/unreadable transcript for {id}: {e}")
            pbar.update(1)
            continue

        for start, end in zip(start_times, end_times):
            vad_list = copy.deepcopy(vad_list_master)

            vads_in_frame_left = [v for v in vad_list[0] if v[1] > start and v[0] < end]
            vads_in_frame_right = [v for v in vad_list[1] if v[1] > start and v[0] < end]

            if vads_in_frame_left:
                vads_in_frame_left[0][0] = max(start, vads_in_frame_left[0][0])
                vads_in_frame_left[-1][1] = min(end, vads_in_frame_left[-1][1])

            if vads_in_frame_right:
                vads_in_frame_right[0][0] = max(start, vads_in_frame_right[0][0])
                vads_in_frame_right[-1][1] = min(end, vads_in_frame_right[-1][1])

            vads_in_frame_left = [[v[0] - start, v[1] - start] for v in vads_in_frame_left]
            vads_in_frame_right = [[v[0] - start, v[1] - start] for v in vads_in_frame_right]

            # future_vad_left = [v for v in vad_list[0] if v[0] > end and v[0] < end + FUTURE_CONTEXT]
            # future_vad_right = [v for v in vad_list[1] if v[0] > end and v[0] < end + FUTURE_CONTEXT]

            # if future_vad_left:
            #     future_vad_left[-1][1] = min(end + FUTURE_CONTEXT, future_vad_left[-1][1])
            #     future_vad_left = [[v[0] - start, v[1] - start] for v in future_vad_left]

            # if future_vad_right:
            #     future_vad_right[-1][1] = min(end + FUTURE_CONTEXT, future_vad_right[-1][1])
            #     future_vad_right = [[v[0] - start, v[1] - start] for v in future_vad_right]

            future_vad_left = copy.deepcopy([v for v in vad_list_master[0] if v[1] > end and v[0] < end + FUTURE_CONTEXT])
            future_vad_right = copy.deepcopy([v for v in vad_list_master[1] if v[1] > end and v[0] < end + FUTURE_CONTEXT])

            # now trim the start and the end
            if future_vad_left != []:
                future_vad_left[0][0] = max(end, future_vad_left[0][0])
                future_vad_left[-1][1] = min(end + FUTURE_CONTEXT, future_vad_left[-1][1])
                future_vad_left = [[v[0] - end, v[1] - end] for v in future_vad_left]

            if future_vad_right != []:
                future_vad_right[0][0] = max(end, future_vad_right[0][0])
                future_vad_right[-1][1] = min(end + FUTURE_CONTEXT, future_vad_right[-1][1])
                future_vad_right = [[v[0] - end, v[1] - end] for v in future_vad_right]

            vad = [vads_in_frame_left, vads_in_frame_right]
            future_vad = [future_vad_left, future_vad_right]

            vad = transcript_utils.vad_list_to_one_hot(vad, TRACK_SIZE_WINDOW, samples=False, sr=AUDIO_FEATURE_HZ, start=0)
            vap = transcript_utils.vad_list_to_one_hot(future_vad, TRACK_SIZE_FUTURE_WINDOW, samples=False, sr=AUDIO_FEATURE_HZ, start=0)

            out_ids.append(id)
            starts.append(start)
            ends.append(end)

            vads = vad.unsqueeze(dim=0) if vads is None else torch.concat((vads, vad.unsqueeze(dim=0)), dim=0)
            vaps = vap.unsqueeze(dim=0) if vaps is None else torch.concat((vaps, vap.unsqueeze(dim=0)), dim=0)

        pbar.update(1)

    out_ids = np.array(out_ids, dtype=np.bytes_)
    starts = np.array(starts, dtype=np.int32)
    ends = np.array(ends, dtype=np.int32)
    return out_ids, starts, ends, vads, vaps


def generate_chunks_for_one_fold(audio_dir, ids, out_prefix):
    out_ids, starts, ends, vads, vaps = get_chunked_dataset(ids, audio_dir)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(out_prefix + "_ids", out_ids)
    np.save(out_prefix + "_starts", starts)
    np.save(out_prefix + "_ends", ends)
    torch.save(vads, out_prefix + "_vads.pt")
    torch.save(vaps, out_prefix + "_vaps.pt")
    print(f"  -> {len(out_ids)} chunk(s) saved to {out_prefix}_{{ids,starts,ends}}.npy / _{{vads,vaps}}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="split directory, e.g. $SRC_AVCOCKTAIL/train")
    parser.add_argument("--out-prefix", required=True, help="output path prefix, e.g. $CHUNKS_OUT_DIR/train")
    parser.add_argument("--csv", default=None, help="optional manifest CSV with an 'id' column; auto-built from --audio-dir if omitted")
    parser.add_argument("--sessions", default=None, help="optional comma-separated list of session names to restrict to (used by the manifest auto-build)")
    args = parser.parse_args()

    sessions = args.sessions.split(",") if args.sessions else None

    if args.csv:
        df = pd.read_csv(args.csv, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        ids = df["id"].tolist()
        if sessions:
            wanted = set(sessions)
            ids = [i for i in ids if i.split("**")[0] in wanted]
        print(f"Loaded {len(ids)} id(s) from {args.csv}")
    else:
        ids = build_manifest(args.audio_dir, sessions=sessions)
        print(f"Auto-built manifest: {len(ids)} id(s) found under {args.audio_dir}")

    print(f"window_size={WINDOW_SIZE}s step_size={STEP_SIZE}s future_context={FUTURE_CONTEXT}s audio_feature_hz={AUDIO_FEATURE_HZ}")
    generate_chunks_for_one_fold(args.audio_dir, ids, args.out_prefix)


if __name__ == "__main__":
    main()

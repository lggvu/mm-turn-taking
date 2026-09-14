"""
Handles speakers whose central_crops face-tracking is split into multiple
tracks (track_00, track_01, ...). Tracks are placed on the central_video.mp4
timeline according to their track_xx.json start_time/end_time, gaps are
filled with black/silent filler, and the heading offset (central.uem.start)
is cut on that aligned timeline -- so a track starting before the offset is
correctly seeked into instead of being dropped or shifted.

Steps: align+cut offset -> upsample to 30fps -> extract wav -> merge L/R
pairs into stereo. Every produced .mp4 / .wav gets a "_fixed" suffix.

Usage:
    python3 process_session_fixed.py --part dev --session session_41
    python3 process_session_fixed.py --part dev --session session_41 \
        --root /home/lhoang/Programming/datasets/mcorec
"""

import argparse
import json
import os
import subprocess

METADATA_NAME = "metadata.json"
LABEL_FOLDER_NAME = "labels"
CLUSTER_LABEL_NAME = "speaker_to_cluster.json"

CROP_W, CROP_H, SRC_FPS, TARGET_FPS, SR = 224, 224, 25, 30, 16000
FIXED_SUFFIX = "_fixed"  # tag appended to the final stereo wav, per this module's docstring


def group_interlocutors(root_folder, part, session_name):
    """
    labels/speaker_to_cluster.json, example:
    {"spk_0": 0, "spk_1": 0, "spk_2": 1, "spk_3": 1, "spk_4": 2, "spk_5": 2}
    returns [[spk_0, spk_1], [spk_2, spk_3], [spk_4, spk_5]]
    """
    label_path = os.path.join(root_folder, part, session_name, LABEL_FOLDER_NAME, CLUSTER_LABEL_NAME)
    with open(label_path) as f:
        speaker_to_cluster = json.load(f)
    cluster_to_speakers = {}
    for spk, cl in speaker_to_cluster.items():
        cluster_to_speakers.setdefault(cl, []).append(spk)
    return list(cluster_to_speakers.values())


def plan_segments(crops, uem_start, uem_end):
    """
    Contiguous list of segments covering [uem_start, uem_end] on the
    central_video timeline. Each segment is either a slice of a real track
    (placed where that track actually sits on the central timeline) or a
    'blank' filler for ranges no track covers.

    Cutting the heading offset falls out for free: we never emit anything
    before uem_start, so a track starting earlier is seeked into by
    (uem_start - track_start) instead of being naively concatenated.
    """
    tracks = sorted(crops, key=lambda c: c["_meta"]["start_time"])
    segments = []
    cursor = uem_start
    for c in tracks:
        m = c["_meta"]
        t_start, t_end = m["start_time"], m["end_time"]
        seg_start = max(t_start, cursor)
        seg_end = min(t_end, uem_end)
        if seg_end - seg_start < 1.0 / SRC_FPS:
            continue  # track lies fully outside the kept window
        if seg_start - cursor >= 1.0 / SRC_FPS:
            segments.append({"type": "blank", "duration": seg_start - cursor})
        segments.append({
            "type": "track",
            "path": c["_path"],
            "seek": seg_start - t_start,
            "duration": seg_end - seg_start,
        })
        cursor = seg_end
        if cursor >= uem_end:
            break
    if uem_end - cursor >= 1.0 / SRC_FPS:
        segments.append({"type": "blank", "duration": uem_end - cursor})
    return segments


def build_aligned_reset(root_folder, part, session, speaker, verbose=True):
    session_dir = os.path.join(root_folder, part, session)
    with open(os.path.join(session_dir, METADATA_NAME)) as f:
        metadata = json.load(f)
    central = metadata[speaker]["central"]
    uem_start, uem_end = central["uem"]["start"], central["uem"]["end"]

    crops = []
    for c in central["crops"]:
        with open(os.path.join(session_dir, c["crop_metadata"])) as f:
            tm = json.load(f)
        crops.append({"_path": os.path.join(session_dir, c["video"]), "_meta": tm})

    segments = plan_segments(crops, uem_start, uem_end)
    if verbose:
        print(f"  {speaker}: uem=[{uem_start}, {uem_end}]  ({len(segments)} segments)")
        for s in segments:
            print("   ", {k: (round(v, 3) if isinstance(v, float) else os.path.basename(v))
                           for k, v in s.items()})

    crops_dir = os.path.dirname(crops[0]["_path"])
    aligned_path = os.path.join(crops_dir, f"{speaker}_central_aligned_reset.mp4")

    # ---- one ffmpeg pass: normalise each segment (real slice or black/silent filler) then concat ----
    inputs, filters, concat_labels = [], [], []
    idx = 0
    for i, seg in enumerate(segments):
        d = f"{seg['duration']:.6f}"
        if seg["type"] == "blank":
            inputs += ["-f", "lavfi", "-t", d, "-i", f"color=c=black:s={CROP_W}x{CROP_H}:r={SRC_FPS}"]
            inputs += ["-f", "lavfi", "-t", d, "-i", f"anullsrc=r={SR}:cl=mono"]
            vsrc, asrc = f"[{idx}:v]", f"[{idx + 1}:a]"
            idx += 2
        else:
            inputs += ["-ss", f"{seg['seek']:.6f}", "-t", d, "-i", seg["path"]]
            vsrc, asrc = f"[{idx}:v]", f"[{idx}:a]"
            idx += 1
        filters.append(f"{vsrc}scale={CROP_W}:{CROP_H},fps={SRC_FPS},setsar=1,format=yuv420p[v{i}]")
        filters.append(f"{asrc}aformat=sample_rates={SR}:channel_layouts=mono,asetpts=PTS-STARTPTS[a{i}]")
        concat_labels.append(f"[v{i}][a{i}]")
    n = len(segments)
    filter_complex = ";".join(filters) + ";" + "".join(concat_labels) + f"concat=n={n}:v=1:a=1[outv][outa]"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", filter_complex,
           "-map", "[outv]", "-map", "[outa]",
           "-r", str(SRC_FPS), "-c:v", "libx264", "-preset", "veryfast",
           "-c:a", "aac", aligned_path]
    subprocess.run(cmd, check=True)

    expected = uem_end - uem_start
    got = float(subprocess.check_output([
        "ffprobe", "-i", aligned_path, "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0"]).decode().strip())
    if abs(got - expected) > 1.0:
        print(f"  WARN {session}/{speaker}: aligned duration {got:.2f}s vs expected {expected:.2f}s")

    # ---- upsample to 30 fps ----
    aligned_30fps = aligned_path.replace(".mp4", "_30fps.mp4")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", aligned_path, "-filter:v", f"fps={TARGET_FPS}", aligned_30fps], check=True)

    # ---- extract wav ----
    aligned_wav = aligned_30fps.replace(".mp4", ".wav")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", aligned_30fps, "-vn", "-ac", "1", "-acodec", "pcm_s16le", aligned_wav], check=True)
    if verbose:
        print(f"   -> {os.path.basename(aligned_wav)}")
    return aligned_wav


def process_session(root_folder, part, session, verbose=True):
    """
    End-to-end: align+cut every speaker in `session`'s 2-speaker groups,
    upsample to 30fps, extract wav, merge each L/R pair into a stereo wav.
    Returns {"wav_by_speaker": {...}, "stereo_paths": [...]}.
    """
    session_dir = os.path.join(root_folder, part, session)
    groups = group_interlocutors(root_folder, part, session)
    pairs = [g for g in groups if len(g) == 2]
    if verbose:
        print(f"==== {part}/{session}  groups={pairs} ====")
    if not pairs:
        print(f"[SKIP] {session}: no 2-speaker groups")
        return {"wav_by_speaker": {}, "stereo_paths": []}

    wav_by_speaker = {}
    for group in pairs:
        for spk in group:
            if spk not in wav_by_speaker:
                wav_by_speaker[spk] = build_aligned_reset(root_folder, part, session, spk, verbose=verbose)

    stereo_dir = os.path.join(session_dir, "central-stereo_audios")
    os.makedirs(stereo_dir, exist_ok=True)
    stereo_paths = []
    for left_spk, right_spk in pairs:
        out_path = os.path.join(stereo_dir, f"{left_spk}-{right_spk}-30fps{FIXED_SUFFIX}.wav")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", wav_by_speaker[left_spk], "-i", wav_by_speaker[right_spk],
                        "-filter_complex", "amerge=inputs=2", "-ac", "2", out_path], check=True)
        if verbose:
            print("  stereo ->", os.path.basename(out_path))
        stereo_paths.append(out_path)

    return {"wav_by_speaker": wav_by_speaker, "stereo_paths": stereo_paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/home/lhoang/Programming/datasets/mcorec", help="dataset root folder")
    parser.add_argument("--part", required=True, help="e.g. dev, train")
    parser.add_argument("--session", required=True, help="e.g. session_41")
    args = parser.parse_args()

    process_session(args.root, args.part, args.session)


if __name__ == "__main__":
    main()

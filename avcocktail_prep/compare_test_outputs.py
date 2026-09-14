"""
Compares the output of test_pipeline.sh (a fresh, isolated run of steps 3+4
on one train session and one dev session) against whatever is already sitting
in the real dataset (--src-root), to sanity-check that the wiring in this
folder's bash scripts reproduces the same audio/video/chunks.

Comparisons:
  - stereo wav: decoded PCM content hash (ffmpeg -> raw s16le -> md5)
  - aligned per-speaker video: decoded frame content hash (ffmpeg -> raw
    yuv420p video-only -> md5), matched to a baseline file by speaker name
    (filenames can differ across pipeline revisions -- see NOTE below)
  - chunks: shapes/counts of the freshly generated ids/starts/ends/vads/vaps

NOTE on naming: process_session.py currently tags the FINAL stereo wav with
"_fixed" unconditionally. Some existing dev sessions already have "_fixed"
baseline data (from an earlier run of an equivalent script), so that
comparison is a real equality check. Train sessions in this dataset snapshot
were previously processed by an older, non-aligned pipeline (no "_fixed"
suffix) -- for those there is no equivalent baseline to diff against, and
this script says so explicitly instead of claiming a false match.
"""

import argparse
import glob
import hashlib
import os
import subprocess

import numpy as np
import torch


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, check=True)


def md5_audio(path):
    """content hash of the decoded PCM samples (ignores container/codec)."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "2", "-ar", "16000", "-"],
        capture_output=True, check=True,
    )
    return hashlib.md5(p.stdout).hexdigest()


def md5_video(path):
    """content hash of the decoded video frames (ignores container/codec)."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-an", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True, check=True,
    )
    return hashlib.md5(p.stdout).hexdigest()


def duration(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, check=True, text=True,
    )
    return float(p.stdout.strip())


def compare_stereo_audio(src_root, test_root, part, session):
    print(f"\n--- stereo audio: {part}/{session} ---")
    src_dir = os.path.join(src_root, part, session, "central-stereo_audios")
    test_dir = os.path.join(test_root, part, session, "central-stereo_audios")

    test_wavs = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    if not test_wavs:
        print("  FAIL: no stereo wav was produced by the test run")
        return

    for test_wav in test_wavs:
        fname = os.path.basename(test_wav)
        src_wav = os.path.join(src_dir, fname)
        dur = duration(test_wav)
        if not os.path.exists(src_wav):
            print(f"  {fname}: NO BASELINE in {src_dir} (nothing to compare; new pipeline output, duration={dur:.2f}s)")
            continue
        same = md5_audio(test_wav) == md5_audio(src_wav)
        same_dur = abs(dur - duration(src_wav)) < 1e-3
        status = "MATCH" if (same and same_dur) else "DIFFERS"
        print(f"  {fname}: {status}  (decoded-PCM match={same}, duration match={same_dur}, {dur:.2f}s)")


def compare_aligned_video(src_root, test_root, part, session):
    print(f"\n--- aligned per-speaker video: {part}/{session} ---")
    test_speakers_dir = os.path.join(test_root, part, session, "speakers")
    src_speakers_dir = os.path.join(src_root, part, session, "speakers")
    if not os.path.isdir(test_speakers_dir):
        print("  FAIL: no speakers/ dir in test output")
        return

    for spk in sorted(os.listdir(test_speakers_dir)):
        test_videos = glob.glob(os.path.join(test_speakers_dir, spk, "central_crops", "*_central_aligned_reset_30fps.mp4"))
        if not test_videos:
            continue
        test_video = test_videos[0]

        # baseline filenames may carry an extra "_fixed" tag from an older
        # pipeline revision -- match by speaker + "_central_aligned_reset" +
        # "_30fps.mp4", not by exact filename.
        baseline_candidates = glob.glob(os.path.join(src_speakers_dir, spk, "central_crops", f"{spk}_central_aligned_reset*_30fps.mp4"))
        if not baseline_candidates:
            print(f"  {spk}: NO BASELINE in {src_speakers_dir}/{spk}/central_crops (new pipeline output, duration={duration(test_video):.2f}s)")
            continue
        baseline = baseline_candidates[0]
        same = md5_video(test_video) == md5_video(baseline)
        same_dur = abs(duration(test_video) - duration(baseline)) < 1e-3
        status = "MATCH" if (same and same_dur) else "DIFFERS"
        print(f"  {spk}: {status}  (decoded-frame match={same}, duration match={same_dur})")
        print(f"        test:     {os.path.basename(test_video)}")
        print(f"        baseline: {os.path.basename(baseline)}")


def compare_chunks(test_chunks_dir, part, session, mm_vap_assets_dir=None):
    print(f"\n--- generated chunks: {part}/{session} ---")
    prefix = os.path.join(test_chunks_dir, part)
    ids_path = prefix + "_ids.npy"
    if not os.path.exists(ids_path):
        print("  FAIL: no chunk output found")
        return

    ids = np.load(ids_path, allow_pickle=True)
    starts = np.load(prefix + "_starts.npy")
    ends = np.load(prefix + "_ends.npy")
    vads = torch.load(prefix + "_vads.pt")
    vaps = torch.load(prefix + "_vaps.pt")
    print(f"  {len(ids)} chunk(s) generated for this session")
    print(f"  starts/ends range: [{starts.min() if len(starts) else '-'}, {ends.max() if len(ends) else '-'}]")
    print(f"  vads shape: {tuple(vads.shape)}  vaps shape: {tuple(vaps.shape)}")

    if not mm_vap_assets_dir:
        return
    # best-effort comparison against a pre-existing baseline chunk asset,
    # filtered to this session. Ids there might not carry the "_fixed" tag,
    # so match by (session, speaker-pair) instead of exact id string.
    old_ids_path = os.path.join(mm_vap_assets_dir, f"{part}_ids.npy")
    if not os.path.exists(old_ids_path):
        print(f"  (no baseline asset found at {old_ids_path} to compare against)")
        return
    old_ids = np.load(old_ids_path, allow_pickle=True)
    old_ids = [x.decode() if isinstance(x, bytes) else x for x in old_ids]
    pair = None
    for i in ids:
        i = i.decode() if isinstance(i, bytes) else i
        s, name = i.split("**")
        pair = name.replace("-30fps", "").replace("_fixed", "")
        break
    n_old_matching = sum(1 for oi in old_ids if oi.startswith(session + "**") and pair in oi)
    print(f"  baseline asset has {n_old_matching} chunk(s) for {session}/{pair} "
          f"vs {len(ids)} freshly generated -- counts may differ if the baseline predates the "
          f"whistle-alignment fix in process_session.py, or used different window/stride settings.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--train-session", default=None, help="omit to skip the train-side comparison")
    ap.add_argument("--dev-session", default=None, help="omit to skip the dev-side comparison")
    ap.add_argument("--baseline-chunks-dir", default=os.environ.get("BASELINE_CHUNKS_DIR"),
                     help="optional dir with pre-existing <part>_ids.npy assets to diff chunk counts against "
                          "(e.g. an older run's output); defaults to $BASELINE_CHUNKS_DIR, comparison is skipped if unset")
    args = ap.parse_args()

    mm_vap_assets_dir = args.baseline_chunks_dir
    test_chunks_dir = os.path.join(args.test_root, "_chunks")

    if args.train_session:
        compare_stereo_audio(args.src_root, args.test_root, "train", args.train_session)
        compare_aligned_video(args.src_root, args.test_root, "train", args.train_session)
        compare_chunks(test_chunks_dir, "train", args.train_session, mm_vap_assets_dir)
    else:
        print("\n(train comparison skipped -- no train session was processed, see warnings above)")

    if args.dev_session:
        compare_stereo_audio(args.src_root, args.test_root, "dev", args.dev_session)
        compare_aligned_video(args.src_root, args.test_root, "dev", args.dev_session)
        compare_chunks(test_chunks_dir, "dev", args.dev_session, mm_vap_assets_dir)
    else:
        print("\n(dev comparison skipped -- no dev session was processed, see warnings above)")


if __name__ == "__main__":
    main()

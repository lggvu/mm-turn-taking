"""
Batch evaluation script for running evaluation on entire dataset.

Usage:
    PYTHONPATH=. python3 eval/evaluate_dataset.py
    PYTHONPATH=. python3 eval/evaluate_dataset.py visualization.enabled=true
"""

import sys
import json
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from datetime import datetime

# Ensure the eval/ directory is first in sys.path so local modules are found
# even when this script is invoked via a symlink or from a different directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_inference import load_model, run_inference, decode_predictions
from evaluator import ShiftHoldEvaluator
from visualization import Visualizer

# manual torch seed
torch.manual_seed(0)

_EVAL_DIR = Path(__file__).resolve().parent


def filter_events_by_time(events: dict, start_time: float, end_time: float) -> dict:
    """
    Filter events to only include those within [start_time, end_time] range.

    Args:
        events: Dict containing 'shifts' and 'holds' with speaker-wise timestamps
        start_time: Start time in seconds (inclusive)
        end_time: End time in seconds (inclusive, -1 means no limit)

    Returns:
        Filtered events dict with same structure
    """
    filtered_events = {'shifts': {}, 'holds': {}}

    # Filter shifts
    for speaker, shifts in events.get('shifts', {}).items():
        filtered_shifts = [
            shift for shift in shifts
            if shift >= start_time and (end_time < 0 or shift <= end_time)
        ]
        if filtered_shifts:
            filtered_events['shifts'][speaker] = filtered_shifts

    # Filter holds
    for speaker, holds in events.get('holds', {}).items():
        filtered_holds = [
            hold for hold in holds
            if hold >= start_time and (end_time < 0 or hold <= end_time)
        ]
        if filtered_holds:
            filtered_events['holds'][speaker] = filtered_holds

    return filtered_events


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file (deprecated - use Hydra cfg instead)."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_session_paths(base_dir: Path, session_id: str, cfg):
    """
    Get paths for a specific session by discovering available event files.

    Args:
        base_dir: Base directory containing all sessions
        session_id: Session identifier (e.g., 'session_00')

    Returns:
        List of dicts with audio_path, events_path, output_dir for each speaker pair
    """
    # session_dir = Path(f"{base_dir}/{session_id}/annotated")
    session_dir = base_dir / session_id
    events_dir = session_dir / "annotated"
    # events_dir = session_dir

    # Find all event files
    event_files = list(Path(events_dir).glob("events*.json"))

    if not event_files:
        return []

    session_paths = []
    for event_file in event_files:
        # Extract speaker pair from event filename
        # Format: events-spk_X_spk_Y.json or events_spk_X_spk_Y.json
        event_name = event_file.stem  # Remove .json

        # Try to extract speaker pair
        import re
        match = re.search(r'(spk_\d+[-_]spk_\d+)', event_name)
        if match:
            # Extract the matched speaker pair and normalize separators
            # Keep 'spk_X' intact, only normalize the separator between speakers to '-'
            speaker_pair_raw = match.group(1)
            # Replace only the middle separator: spk_X_spk_Y -> spk_X-spk_Y
            speaker_pair = speaker_pair_raw.replace('_spk_', '-spk_')

            audio_path = session_dir / "stereo_audios" / f"{speaker_pair}-30fps.wav"

            # Use results directory relative to this eval/ folder
            dataset_name = cfg.data.get('name', 'unknown')
            results_base_dir = _EVAL_DIR / "results"
            output_dir = results_base_dir / dataset_name / cfg.prefix_save / session_id / speaker_pair

            session_paths.append({
                'audio_path': str(audio_path),
                'events_path': str(event_file),
                'output_dir': str(output_dir),
                'speaker_pair': speaker_pair
            })

    return session_paths


def evaluate_session(
    model,
    session_paths: dict,
    cfg: dict,
    video_dir: str,
    channelmap: str
):
    """
    Evaluate a single session.

    Returns:
        results: Dict with evaluation metrics
        y_true: Ground truth labels
        y_pred: Predicted labels
    """
    # Check if files exist
    audio_path = Path(session_paths['audio_path'])
    events_path = Path(session_paths['events_path'])

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return None, None, None

    if not events_path.exists():
        print(f"Events file not found: {events_path}")
        return None, None, None

    # Create output directory
    output_dir = Path(session_paths['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth events first to extract timestamps
    events = json.load(open(events_path, 'r'))
    event_data = events.get(cfg.evaluation.event_type, events)

    # Extract event timestamps for embedding extraction
    event_timestamps = []
    for event_type in ['shifts', 'holds']:
        if event_type in event_data:
            for speaker, timestamps in event_data[event_type].items():
                for ts in timestamps:
                    event_timestamps.append((event_type[:-1], ts))  # 'shifts' -> 'shift'

    # Set event timestamps on model for embedding extraction
    if hasattr(model, 'set_event_timestamps'):
        model.set_event_timestamps(event_timestamps, window_sec=0.5)
        print(f"  Set {len(event_timestamps)} events for embedding extraction")

    # try:
    # Run inference
    vaps, vads = run_inference(
        model=model,
        audio_path=str(audio_path),
        video_dir=video_dir,
        channelmap=channelmap,
        transcript_path=None,
        window_size=cfg.inference.window_size,
        step_size=cfg.inference.step_size,
        mode=cfg.inference.mode,
        feature_sr=cfg.inference.feature_sr
    )

    # Decode predictions
    print(f'cfg.inference.mode: {cfg.inference.mode}')
    predictions = decode_predictions(vaps, mode=cfg.inference.mode)
    p_future = predictions['p_future']

    # Retrieve and save accumulated embeddings around events
    if hasattr(model, 'get_accumulated_embeddings'):
        accumulated_embeddings = model.get_accumulated_embeddings()
        if accumulated_embeddings:
            print(f"  Collected {len(accumulated_embeddings)} embedding frames around events")

            # Convert to serializable format (detach tensors)
            embeddings_data = []
            for item in accumulated_embeddings:
                embeddings_data.append({
                    'event_type': item['event_type'],
                    'event_time': float(item['event_time']),
                    'absolute_time': float(item['absolute_time']),
                    'relative_time': float(item['relative_time']),
                    'embedding': item['embedding'].squeeze(0).numpy().tolist(),  # [256]
                    'chunk_idx': int(item['chunk_idx'])
                })

            if cfg.save_embs:
                import pickle
                embeddings_file = output_dir / f"{cfg.prefix_save}_embeddings.pkl"
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(embeddings_data, f)
                print(f"  Saved embeddings to {embeddings_file}")

            # Reset for next session
            model.reset_embedding_extraction()

    # events already loaded above, just use the filtered version

    # Filter events by evaluation time range
    if cfg.evaluation.eval_start > 0 or cfg.evaluation.eval_end >= 0:
        event_data = filter_events_by_time(
            event_data,
            cfg.evaluation.eval_start,
            cfg.evaluation.eval_end
        )

    # Evaluate
    evaluator = ShiftHoldEvaluator(feature_sr=cfg.inference.feature_sr)
    y_true, y_pred, y_pred_w_time, results = evaluator.evaluate(
        events=event_data,
        p_future=p_future,
        window=cfg.evaluation.window,
        threshold=cfg.evaluation.threshold,
        debug=False,  # Don't print debug for batch processing
        output_dir=str(output_dir)
    )

    # Save raw predictions
    with open(output_dir / f"{cfg.prefix_save}_raw_predictions.json", "w") as f:
        json.dump(y_pred_w_time, f, indent=2)

    # Save results
    with open(output_dir / f"{cfg.prefix_save}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create visualizations if enabled
    if cfg.visualization.enabled:
        visualizer = Visualizer(output_dir=str(output_dir))

        # Detailed comparison plot
        visualizer.plot_detailed_comparison(
            p_future=p_future,
            vads=vads,
            events=event_data,
            y_true=y_true,
            y_pred=y_pred,
            audio_path=str(audio_path),
            feature_sr=cfg.inference.feature_sr,
            start_time=cfg.visualization.start_time,
            stop_time=cfg.visualization.stop_time
        )

        # Confusion matrix
        visualizer.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred
        )

    return results, y_true, y_pred

    # except Exception as e:
    #     print(f"  Error: {e}")
    #     return None, None, None


def aggregate_results(all_results: list):
    """
    Aggregate results across all sessions.

    Args:
        all_results: List of (session_id, results, y_true, y_pred) tuples

    Returns:
        aggregated: Metrics computed on concatenated predictions
        averaged: Average of per-session metrics
    """
    # Filter out None results
    valid_results = [(sid, r, yt, yp) for sid, r, yt, yp in all_results
                     if r is not None and yt is not None and yp is not None]

    if not valid_results:
        return None, None

    # Concatenate all predictions for aggregated metrics
    all_y_true = np.concatenate([yt for _, _, yt, _ in valid_results])
    all_y_pred = np.concatenate([yp for _, _, _, yp in valid_results])

    # Compute aggregated metrics
    from sklearn.metrics import f1_score, precision_score, recall_score

    f1_per_class = f1_score(all_y_true, all_y_pred, labels=[0, 1], average=None, zero_division=0)
    weighted_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    precision_per_class = precision_score(all_y_true, all_y_pred, labels=[0, 1], average=None, zero_division=0)
    recall_per_class = recall_score(all_y_true, all_y_pred, labels=[0, 1], average=None, zero_division=0)

    aggregated = {
        'hold_f1': float(f1_per_class[0]),
        'shift_f1': float(f1_per_class[1]),
        'weighted_f1': float(weighted_f1),
        'hold_precision': float(precision_per_class[0]),
        'shift_precision': float(precision_per_class[1]),
        'hold_recall': float(recall_per_class[0]),
        'shift_recall': float(recall_per_class[1]),
        'n_holds': int(np.sum(all_y_true == 0)),
        'n_shifts': int(np.sum(all_y_true == 1)),
        'n_sessions': len(valid_results)
    }

    # Compute averaged metrics (mean across sessions)
    metric_keys = ['hold_f1', 'shift_f1', 'weighted_f1',
                   'hold_precision', 'shift_precision',
                   'hold_recall', 'shift_recall']

    averaged = {
        key: float(np.mean([r[key] for _, r, _, _ in valid_results]))
        for key in metric_keys
    }
    averaged['n_sessions'] = len(valid_results)
    averaged['total_holds'] = int(np.sum([r['n_holds'] for _, r, _, _ in valid_results]))
    averaged['total_shifts'] = int(np.sum([r['n_shifts'] for _, r, _, _ in valid_results]))

    return aggregated, averaged


def print_results(title: str, results: dict):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    print(f"{'Metric':<25} {'Shift':<15} {'Hold':<15}")
    print("-"*70)
    print(f"{'F1 Score':<25} {results['shift_f1']:<15.4f} {results['hold_f1']:<15.4f}")
    print(f"{'Precision':<25} {results['shift_precision']:<15.4f} {results['hold_precision']:<15.4f}")
    print(f"{'Recall':<25} {results['shift_recall']:<15.4f} {results['hold_recall']:<15.4f}")
    print("-"*70)
    print(f"{'Weighted F1':<25} {results['weighted_f1']:<15.4f}")

    if 'n_shifts' in results and 'n_holds' in results:
        print(f"{'Count':<25} {results['n_shifts']:<15} {results['n_holds']:<15}")
    elif 'total_shifts' in results and 'total_holds' in results:
        print(f"{'Total Count':<25} {results['total_shifts']:<15} {results['total_holds']:<15}")

    if 'n_sessions' in results:
        print(f"{'Sessions':<25} {results['n_sessions']}")

    print("="*70 + "\n")


@hydra.main(config_path="conf", config_name="config_avcocktail")
def main(cfg: DictConfig):
    """Main batch evaluation pipeline."""
    # Hydra changes the process cwd to its run dir (outputs/<date>/<time>) on
    # some versions (e.g. 1.0.x), so relative CLI overrides like
    # model.weights_path=../save/... must be resolved against the original
    # launch directory, not the current one.
    if cfg.model.config_path:
        cfg.model.config_path = hydra.utils.to_absolute_path(cfg.model.config_path)
    if cfg.model.weights_path:
        cfg.model.weights_path = hydra.utils.to_absolute_path(cfg.model.weights_path)

    # # Auto-generate result name from checkpoint path if not manually specified
    # if not cfg.prefix_save:
    #     weights_path = cfg.model.weights_path
    #     if weights_path:
    #         # Extract checkpoint name from path
    #         if "save" in weights_path:
    #             # anything after "save" is the checkpoint name
    #             checkpoint_name = weights_path.split("save")[-1].strip("/").replace('/', '-')
    #         else:
    #             checkpoint_name = Path(weights_path).stem
    #         # Replace / with - to avoid creating subfolders
    #         checkpoint_name = checkpoint_name.replace('/', '-')
    #         # Generate name: <dataset>-<checkpoint>
    #         dataset_name = cfg.data.get('name', 'unknown')
    #         cfg.prefix_save = f"{dataset_name}-{checkpoint_name}"
    #     else:
    #         cfg.prefix_save = f"{cfg.data.get('name', 'unknown')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not cfg.prefix_save:
        weights_path = cfg.model.weights_path
        if weights_path:
            if "save" in weights_path:
                checkpoint_name = weights_path.split("save")[-1].strip("/").replace('/', '-')
            else:
                checkpoint_name = Path(weights_path).stem
            dataset_name = cfg.data.get('name', 'unknown')
            cfg.prefix_save = f"{dataset_name}_test-{checkpoint_name}"
        else:
            cfg.prefix_save = f"{cfg.data.get('name', 'unknown')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Setup results directory
    dataset_name = cfg.data.get('name', 'unknown')
    results_base_dir = _EVAL_DIR / "results"
    results_dir = results_base_dir / dataset_name / cfg.prefix_save
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config_file': str(Path(cfg.model.config_path).resolve()) if cfg.model.config_path else None,
        'weights_file': str(Path(cfg.model.weights_path).resolve()) if cfg.model.weights_path else None,
        'dataset_name': dataset_name,
        'result_name': cfg.prefix_save,
        'full_config': OmegaConf.to_container(cfg, resolve=True)
    }

    with open(results_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults will be saved to: {results_dir}")
    print(f"Result name: {cfg.prefix_save}\n")

    # Sessions to evaluate, and the directory they live under -- both come from
    # cfg.data (see eval/conf/config_avcocktail.yaml), not hardcoded here, so a
    # different split/host can be evaluated purely via config/CLI overrides.
    sessions = sorted(cfg.data.sessions)
    base_dir = Path(cfg.data.video_dir)

    print("="*70)
    print("BATCH EVALUATION: Shift/Hold Prediction")
    print("="*70)
    print(f"Sessions to evaluate: {len(sessions)}")
    print(f"Base directory: {base_dir}")
    print(f"Event type: {cfg.evaluation.event_type}")
    print(f"Threshold: {cfg.evaluation.threshold}")
    print(f"Window: {cfg.evaluation.window}s")
    print(f"Eval region: [{cfg.evaluation.eval_start}s, {cfg.evaluation.eval_end}s]")
    print("="*70 + "\n")

    # Load model
    print("\n1. Loading model...")
    model_name = cfg.model.model_name
    valid_model_names = [
        "early_fusion_candor",
        "vap_candor",
        "vap_switchboard_asr",
        "vap_switchboard_ground_truth",
        "video_candor",
        "vap",
        "video",
        "early_fusion",
        "late_fusion"
    ]
    if model_name not in valid_model_names:
        raise ValueError(f"Invalid model name '{model_name}'. Must be one of {valid_model_names}")
    else:
        if not cfg.model.config_path and not cfg.model.weights_path:
            raise ValueError(
                f"No config_path or weights_path provided for model '{model_name}'. "
                "Please pass model.config_path and model.weights_path on the command line."
            )
        else:
            config_path = cfg.model.config_path
            weights_path = cfg.model.weights_path
    model = load_model(
            model_name,
            config_path=config_path,
            weights_path=weights_path,
            device=cfg.model.device
    )
    print("Model loaded\n")

    # Evaluate each session
    all_results = []

    for session_id in tqdm(sessions, desc="Evaluating sessions"):
        print(f"\n{session_id}:")

        # Get all speaker pairs for this session
        session_paths_list = get_session_paths(base_dir, session_id, cfg)

        if not session_paths_list:
            print(f"  No event files found")
            continue

        for session_paths in session_paths_list:
            speaker_pair = session_paths['speaker_pair']
            print(f"  {speaker_pair}:")

            # Evaluate
            results, y_true, y_pred = evaluate_session(
                model=model,
                session_paths=session_paths,
                cfg=cfg,
                video_dir=cfg.data.video_dir,
                channelmap=cfg.data.channelmap
            )

            if results is not None:
                print(f"    F1: {results['weighted_f1']:.4f} | "
                      f"Shifts: {results['n_shifts']} | Holds: {results['n_holds']}")
                all_results.append((f"{session_id}/{speaker_pair}", results, y_true, y_pred))
            else:
                print(f"    Evaluation failed")

    # Aggregate results
    print("\n" + "="*70)
    print("COMPUTING AGGREGATE STATISTICS")
    print("="*70)

    aggregated, averaged = aggregate_results(all_results)

    if aggregated is not None and averaged is not None:
        # Print aggregated results (all predictions concatenated)
        print_results("AGGREGATED RESULTS (All Predictions)", aggregated)

        # Print averaged results (mean across sessions)
        print_results("AVERAGED RESULTS (Mean Across Sessions)", averaged)

        # Save aggregate results
        dataset_name = cfg.data.get('name', 'unknown')
        results_base_dir = _EVAL_DIR / "results"
        results_dir = results_base_dir / dataset_name / cfg.prefix_save
        output_file = results_dir / f"aggregate_results_{cfg.evaluation.window}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'sessions_evaluated': [sid for sid, _, _, _ in all_results],
                'n_sessions': len(all_results),
                'aggregated': aggregated,
                'averaged': averaged,
                'per_session': {
                    sid: r for sid, r, _, _ in all_results
                }
            }, f, indent=2)

        print(f"Aggregate results saved to: {output_file}")
    else:
        print("No valid results to aggregate")

    print("\n" + "="*70)
    print("BATCH EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

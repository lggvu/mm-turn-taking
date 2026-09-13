"""
Batch evaluation script for running evaluation on CANDOR dataset.
# TODO havent checked if it produces the same results!
Usage:
    cd eval
    PYTHONPATH=.. python evaluate_candor_folds.py
    PYTHONPATH=.. python evaluate_candor_folds.py visualization.enabled=true
"""
import sys
import os
import json
import csv
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure the eval/ directory is first in sys.path so local modules are found
# even when this script is invoked via a symlink or from a different directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_inference import load_model, run_inference, decode_predictions
from evaluator import ShiftHoldEvaluator
from visualization import Visualizer

torch.manual_seed(0)

_EVAL_DIR = Path(__file__).resolve().parent


def filter_events_by_time(events: dict, start_time: float, end_time: float) -> dict:
    filtered_events = {'shifts': {}, 'holds': {}}
    for speaker, shifts in events.get('shifts', {}).items():
        filtered = [s for s in shifts if s >= start_time and (end_time < 0 or s <= end_time)]
        if filtered:
            filtered_events['shifts'][speaker] = filtered
    for speaker, holds in events.get('holds', {}).items():
        filtered = [h for h in holds if h >= start_time and (end_time < 0 or h <= end_time)]
        if filtered:
            filtered_events['holds'][speaker] = filtered
    return filtered_events


def load_fold_sessions(fold_dir: Path, fold_num: int) -> list:
    val_csv = fold_dir / f"fold_{fold_num}" / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation file not found: {val_csv}")
    sessions = []
    with open(val_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append(row['id'])
    return sessions


def get_session_paths(base_dir: Path, session_id: str, cfg: DictConfig) -> list:
    audio_path = base_dir / "candor_wav" / f"{session_id}.wav"
    events_path = base_dir / cfg.data.events_dir / f"{session_id}.json"
    video_path = base_dir / "candor_openface_pkl" / f"{session_id}.pkl"

    if not audio_path.exists() or not events_path.exists():
        print(f'Cannot find {audio_path}')
        return []

    return [{
        'audio_path': str(audio_path),
        'events_path': str(events_path),
        'video_path': str(video_path),
        'output_dir': str(base_dir / f"evaluation_results_{cfg.evaluation.eval_start}" / session_id),
        'speaker_pair': session_id,
    }]


def evaluate_session(model, session_paths: dict, cfg: DictConfig, video_dir: str, channelmap_path: str):
    audio_path = Path(session_paths['audio_path'])
    events_path = Path(session_paths['events_path'])

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return None, None, None
    if not events_path.exists():
        print(f"Events file not found: {events_path}")
        return None, None, None

    output_dir = Path(session_paths['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    vaps, vads = run_inference(
        model=model,
        audio_path=str(audio_path),
        video_dir=video_dir,
        channelmap=channelmap_path,
        transcript_path=None,
        window_size=cfg.inference.window_size,
        step_size=cfg.inference.step_size,
        mode=cfg.inference.mode,
        feature_sr=cfg.inference.feature_sr,
        data="candor"
    )

    print(f'cfg.inference.mode: {cfg.inference.mode}')
    predictions = decode_predictions(vaps, mode=cfg.inference.mode)
    p_future = predictions['p_future']
    print(f'p_future: {p_future}')

    events = json.load(open(events_path, 'r'))
    event_data = events.get(cfg.evaluation.event_type, events)

    if cfg.evaluation.eval_start > 0 or cfg.evaluation.eval_end >= 0:
        event_data = filter_events_by_time(event_data, cfg.evaluation.eval_start, cfg.evaluation.eval_end)

    evaluator = ShiftHoldEvaluator(feature_sr=cfg.inference.feature_sr)
    y_true, y_pred, y_pred_w_time, results = evaluator.evaluate(
        events=event_data,
        p_future=p_future,
        window=cfg.evaluation.window,
        threshold=cfg.evaluation.threshold,
        debug=False,
        output_dir=str(output_dir)
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / f"{cfg.prefix_save}_raw_predictions.json", "w+") as f:
        json.dump(y_pred_w_time, f, indent=2)
    with open(output_dir / f"{cfg.prefix_save}_results.json", "w+") as f:
        json.dump(results, f, indent=2)

    if cfg.visualization.enabled:
        visualizer = Visualizer(output_dir=str(output_dir))
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
        visualizer.plot_confusion_matrix(y_true=y_true, y_pred=y_pred)

    return results, y_true, y_pred


def aggregate_results(all_results: list):
    valid_results = [(sid, r, yt, yp) for sid, r, yt, yp in all_results
                     if r is not None and yt is not None and yp is not None]
    if not valid_results:
        return None, None

    all_y_true = np.concatenate([yt for _, _, yt, _ in valid_results])
    all_y_pred = np.concatenate([yp for _, _, _, yp in valid_results])

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

    metric_keys = ['hold_f1', 'shift_f1', 'weighted_f1',
                   'hold_precision', 'shift_precision',
                   'hold_recall', 'shift_recall']
    averaged = {key: float(np.mean([r[key] for _, r, _, _ in valid_results])) for key in metric_keys}
    averaged['n_sessions'] = len(valid_results)
    averaged['total_holds'] = int(np.sum([r['n_holds'] for _, r, _, _ in valid_results]))
    averaged['total_shifts'] = int(np.sum([r['n_shifts'] for _, r, _, _ in valid_results]))

    return aggregated, averaged


def compute_average_across_folds(fold_results: dict) -> dict:
    metric_keys = ['hold_f1', 'shift_f1', 'weighted_f1',
                   'hold_precision', 'shift_precision',
                   'hold_recall', 'shift_recall']

    aggregated_results = [agg for agg, _ in fold_results.values()]

    avg_across_folds = {
        key: float(np.mean([r[key] for r in aggregated_results]))
        for key in metric_keys
    }
    avg_across_folds.update({
        f'{key}_std': float(np.std([r[key] for r in aggregated_results]))
        for key in metric_keys
    })
    avg_across_folds['n_folds'] = len(fold_results)
    avg_across_folds['total_sessions'] = sum(agg['n_sessions'] for agg, _ in fold_results.values())
    return avg_across_folds


def print_results(title: str, results: dict, show_std: bool = False):
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    print(f"{'Metric':<25} {'Shift':<15} {'Hold':<15}")
    print("-"*70)

    if show_std:
        print(f"{'F1 Score':<25} {results['shift_f1']:<15.4f} {results['hold_f1']:<15.4f}")
        print(f"{'  (± std)':<25} {results['shift_f1_std']:<15.4f} {results['hold_f1_std']:<15.4f}")
        print(f"{'Precision':<25} {results['shift_precision']:<15.4f} {results['hold_precision']:<15.4f}")
        print(f"{'  (± std)':<25} {results['shift_precision_std']:<15.4f} {results['hold_precision_std']:<15.4f}")
        print(f"{'Recall':<25} {results['shift_recall']:<15.4f} {results['hold_recall']:<15.4f}")
        print(f"{'  (± std)':<25} {results['shift_recall_std']:<15.4f} {results['hold_recall_std']:<15.4f}")
        print("-"*70)
        print(f"{'Weighted F1':<25} {results['weighted_f1']:<15.4f}")
        print(f"{'  (± std)':<25} {results['weighted_f1_std']:<15.4f}")
    else:
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
    elif 'total_sessions' in results:
        print(f"{'Total Sessions':<25} {results['total_sessions']}")

    if 'n_folds' in results:
        print(f"{'Folds':<25} {results['n_folds']}")

    print("="*70 + "\n")


@hydra.main(config_path="conf", config_name="config_candor")
def main(cfg: DictConfig):
    """Main batch evaluation pipeline with fold support."""

    if not cfg.prefix_save:
        weights_path = cfg.model.weights_path
        if weights_path:
            if "checkpoints" in weights_path:
                checkpoint_name = Path(weights_path).stem
            else:
                checkpoint_name = Path(weights_path).stem
            checkpoint_name = checkpoint_name.replace('/', '-')
            dataset_name = cfg.data.get('name', 'unknown')
            cfg.prefix_save = f"{dataset_name}-{checkpoint_name}"
        else:
            cfg.prefix_save = f"{cfg.data.get('name', 'unknown')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    dataset_name = cfg.data.get('name', 'unknown')
    results_base_dir = Path(cfg.evaluation.results_dir) if cfg.evaluation.get('results_dir') else _EVAL_DIR / "results"

    results_dir = results_base_dir / dataset_name / cfg.prefix_save
    results_dir.mkdir(parents=True, exist_ok=True)

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

    fold_dir = Path(cfg.data.fold_dir)

    if cfg.evaluation.fold == "all":
        folds_to_evaluate = list(range(5))
    else:
        folds_to_evaluate = [int(cfg.evaluation.fold)]

    base_dir = Path(cfg.data.base_dir)

    print("="*70)
    print("BATCH EVALUATION: Shift/Hold Prediction (Fold-based)")
    print("="*70)
    print(f"Folds to evaluate: {folds_to_evaluate}")
    print(f"Base directory: {base_dir}")
    print(f"Event type: {cfg.evaluation.event_type}")
    print(f"Threshold: {cfg.evaluation.threshold}")
    print(f"Window: {cfg.evaluation.window}s")
    print(f"Eval region: [{cfg.evaluation.eval_start}s, {cfg.evaluation.eval_end}s]")
    print("="*70 + "\n")

    print("\n1. Loading model...")
    model = load_model(
        cfg.model.model_name,
        config_path=cfg.model.config_path,
        weights_path=cfg.model.weights_path,
        device=cfg.model.device
    )
    print("Model loaded\n")

    fold_results = {}

    for fold_num in folds_to_evaluate:
        print("\n" + "#"*70)
        print(f"EVALUATING FOLD {fold_num}")
        print("#"*70)

        sessions = sorted(load_fold_sessions(fold_dir, fold_num))
        print(f"Sessions in fold {fold_num}: {len(sessions)}\n")

        all_results = []

        for session_id in tqdm(sessions, desc=f"Evaluating fold {fold_num}"):
            print(f"session_id={session_id}:")

            session_paths_list = get_session_paths(base_dir, session_id, cfg)

            if not session_paths_list:
                print(f"  No event files found")
                continue

            for session_paths in session_paths_list:
                try:
                    results, y_true, y_pred = evaluate_session(
                        model=model,
                        session_paths=session_paths,
                        cfg=cfg,
                        video_dir=cfg.data.video_dir,
                        channelmap_path=cfg.data.channelmap,
                    )
                except ValueError:
                    print(f"Evaluation skipped: {session_id}")
                    continue

                if results is not None:
                    print(f"  F1: {results['weighted_f1']:.4f} | "
                          f"Shifts: {results['n_shifts']} | Holds: {results['n_holds']}")
                    all_results.append((session_id, results, y_true, y_pred))
                else:
                    print(f"  Evaluation failed")

        print("\n" + "="*70)
        print(f"COMPUTING STATISTICS FOR FOLD {fold_num}")
        print("="*70)

        aggregated, averaged = aggregate_results(all_results)

        if aggregated is not None and averaged is not None:
            fold_results[fold_num] = (aggregated, averaged)

            print_results(f"FOLD {fold_num} - AGGREGATED RESULTS (All Predictions)", aggregated)
            print_results(f"FOLD {fold_num} - AVERAGED RESULTS (Mean Across Sessions)", averaged)

            output_file = results_dir / f"fold_{fold_num}_aggregate_results.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'fold': fold_num,
                    'sessions_evaluated': [sid for sid, _, _, _ in all_results],
                    'n_sessions': len(all_results),
                    'aggregated': aggregated,
                    'averaged': averaged,
                    'per_session': {sid: r for sid, r, _, _ in all_results}
                }, f, indent=2)

            print(f"Fold {fold_num} results saved to: {output_file}")
        else:
            print(f"No valid results for fold {fold_num}")

    if len(folds_to_evaluate) > 1 and len(fold_results) > 0:
        print("\n" + "="*70)
        print("COMPUTING AVERAGE ACROSS FOLDS")
        print("="*70)

        avg_across_folds = compute_average_across_folds(fold_results)
        print_results("AVERAGE ACROSS ALL FOLDS", avg_across_folds, show_std=True)

        output_file = results_dir / "average_across_folds.json"
        with open(output_file, 'w') as f:
            json.dump({
                'folds_evaluated': list(fold_results.keys()),
                'average_metrics': avg_across_folds,
                'per_fold_aggregated': {f'fold_{k}': v[0] for k, v in fold_results.items()},
                'per_fold_averaged': {f'fold_{k}': v[1] for k, v in fold_results.items()}
            }, f, indent=2)

        print(f"Average across folds results saved to: {output_file}")

    print("\n" + "="*70)
    print("BATCH EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

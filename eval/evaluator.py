"""
Evaluation utilities for shift/hold predictions.
"""

import numpy as np
import torch
from typing import Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
import random

class ShiftHoldEvaluator:
    """Evaluates shift/hold predictions using VAP probabilities."""
    
    def __init__(self, feature_sr: int = 50):
        """
        Args:
            feature_sr: Feature sampling rate in Hz
        """
        self.feature_sr = feature_sr
    
    def evaluate(
        self, 
        events: Dict, 
        p_future: torch.Tensor, 
        window: float = 0.2,
        threshold: float = 0.5,
        debug: bool = False,
        output_dir: str = "."
    ) -> Dict[str, float]:
        """
        Evaluate shift/hold predictions.
        
        Uses the validation.py method:
        - For each event, extract a window of predictions
        - Sum the other speaker's probabilities in that window as the score
        - High score = shift predicted (other speaker will talk)
        - Low score = hold predicted (current speaker continues)
        
        Args:
            events: Dict containing 'shifts' and 'holds' with speaker-wise timestamps
            p_future: Tensor of shape [T, 2] with per-speaker probabilities
            window: Time window in seconds around each event
            threshold: Score threshold for predicting shift
            debug: If True, print wrongly classified timestamps
            output_dir: Directory to save predictions log
        
        Returns:
            Dict with metrics: shift_f1, hold_f1, weighted_f1, shift_precision, 
            shift_recall, hold_precision, hold_recall, n_shifts, n_holds
        """
        # Get ground truth labels and scores
        y_true, scores, scores_w_time = self._get_shifts_holds(events, p_future, window)
        
        if len(y_true) == 0:
            return self._empty_results()
        
        # Apply threshold to get predictions
        # score >= threshold means shift (y_pred=1), otherwise hold (y_pred=0)
        y_pred = np.array([1 if score >= threshold else 0 for score in scores])


        y_true = np.array(y_true)
        y_pred_w_time = {time: 1 if score >= threshold else 0 for time, score in scores_w_time.items()}
        
        # Print misclassifications if debug mode
        if debug:
            self._print_misclassifications(y_true, y_pred, scores, events, window)
            self._log_predictions(y_true, y_pred, scores, events, output_dir)
        
        return y_true, y_pred, y_pred_w_time, self._compute_metrics(y_true, y_pred)
    
    def _samples(self, time: float) -> int:
        """Convert time in seconds to sample index."""
        return int(time * self.feature_sr)
    
    def _get_shifts_holds(
        self, 
        events: Dict, 
        p: torch.Tensor, 
        window: float = 0.2
    ) -> Tuple[list, list]:
        """
        Extract ground truth labels and prediction scores for shift/hold events.
        
        This follows the validation.py logic:
        - For each shift: label=1, score=sum of OTHER speaker's probability in window
        - For each hold: label=0, score=sum of OTHER speaker's probability in window
        
        Args:
            events: Dict with 'shifts' and 'holds' keys
            p: Tensor of shape [T, 2] with per-speaker probabilities
            window: Time window in seconds (positive = forward, negative = backward)
        
        Returns:
            y_true: List of ground truth labels (1=shift, 0=hold)
            scores: List of prediction scores (higher = more likely shift)
        """
        # Convert window to samples
        if window < 0:
            sign = -1
        else:
            sign = 1
        window_samples = self._samples(np.abs(window)) * sign
        
        max_time = p.shape[0] / self.feature_sr
        p_np = p.cpu().numpy()
        
        y_true = []
        scores = []
        scores_w_time = {}
        
        # Process shift events (label = 1)
        shift_events = events.get('shifts', {})
        for speaker, shifts in shift_events.items():
            for shift in shifts:
                if shift > max_time:
                    continue
                
                speaker = int(speaker)
                
                # Define window around the event
                end_original, start_original = self._samples(shift), self._samples(shift) + window_samples
                start = min(end_original, start_original)
                end = max(end_original, start_original)
                
                # Sum the OTHER speaker's probability in the window
                next_speaker_prob = p_np[start:end, 1-speaker].sum()
                
                y_true.append(1)  # shift
                scores.append(float(next_speaker_prob))
                scores_w_time[shift] = float(next_speaker_prob)
        
        # Process hold events (label = 0)
        hold_events = events.get('holds', {})
        for speaker, holds in hold_events.items():
            for hold in holds:
                if hold > max_time:
                    continue
                
                speaker = int(speaker)
                
                # Define window around the event
                end_original, start_original = self._samples(hold), self._samples(hold) + window_samples
                start = min(end_original, start_original)
                end = max(end_original, start_original)
                
                # Sum the OTHER speaker's probability in the window
                next_speaker_prob = p_np[start:end, 1-speaker].sum()
                
                y_true.append(0)  # hold
                scores.append(float(next_speaker_prob))
                scores_w_time[hold] = float(next_speaker_prob)
        print(f'scores: {scores}')
        return y_true, scores, scores_w_time
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        # F1 scores
        f1_per_class = f1_score(
            y_true, y_pred, 
            labels=[0, 1], 
            average=None, 
            zero_division=0
        )
        weighted_f1 = f1_score(
            y_true, y_pred, 
            average='weighted', 
            zero_division=0
        )
        
        # Precision scores
        precision_per_class = precision_score(
            y_true, y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0
        )
        
        # Recall scores
        recall_per_class = recall_score(
            y_true, y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0
        )
        
        return {
            'hold_f1': float(f1_per_class[0]),
            'shift_f1': float(f1_per_class[1]),
            'weighted_f1': float(weighted_f1),
            'hold_precision': float(precision_per_class[0]),
            'shift_precision': float(precision_per_class[1]),
            'hold_recall': float(recall_per_class[0]),
            'shift_recall': float(recall_per_class[1]),
            'n_holds': int(np.sum(y_true == 0)),
            'n_shifts': int(np.sum(y_true == 1))
        }
    
    def _print_misclassifications(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: list,
        events: Dict,
        window: float
    ):
        """Print wrongly classified timestamps with scores for debugging."""
        misclassified = y_true != y_pred
        
        if not misclassified.any():
            print("\n✓ No misclassifications!")
            return
        
        print("\n" + "="*80)
        print("MISCLASSIFIED EVENTS")
        print("="*80)
        
        label_names = {0: "HOLD", 1: "SHIFT"}
        
        # Reconstruct event list with timestamps (must match order in _get_shifts_holds)
        event_list = []
        max_time = len(y_true) / self.feature_sr  # Approximate max time
        
        # Process shifts first (same order as in _get_shifts_holds)
        for speaker, shifts in events.get('shifts', {}).items():
            for shift in shifts:
                event_list.append((shift, 1, int(speaker)))  # (time, label, speaker)
        
        # Then process holds (same order as in _get_shifts_holds)
        for speaker, holds in events.get('holds', {}).items():
            for hold in holds:
                event_list.append((hold, 0, int(speaker)))  # (time, label, speaker)
        
        # DO NOT sort - keep the same order as _get_shifts_holds builds y_true/y_pred
        
        misclassified_indices = np.where(misclassified)[0]
        for idx in misclassified_indices:
            if idx < len(event_list):
                time, true_label, speaker = event_list[idx]
                pred_label = y_pred[idx]
                score = scores[idx]
            else:
                continue  # Skip if index out of bounds
            
            print(f"Time: {time:7.2f}s | Speaker: {speaker} | "
                  f"True: {label_names[true_label]:5s} | "
                  f"Pred: {label_names[pred_label]:5s} | "
                  f"Score: {score:6.3f} | "
                  f"ERROR: {label_names[pred_label]} predicted instead of {label_names[true_label]}")
        
        # Summary statistics
        n_misclassified = misclassified.sum()
        n_total = len(y_true)
        accuracy = (n_total - n_misclassified) / n_total * 100
        
        # Count error types (note: labels are 1=shift, 0=hold)
        shift_as_hold = np.sum((y_true == 1) & (y_pred == 0))
        hold_as_shift = np.sum((y_true == 0) & (y_pred == 1))
        
        print("="*80)
        print(f"Total misclassifications: {n_misclassified}/{n_total} ({100 - accuracy:.1f}% error rate)")
        print(f"  - SHIFT predicted as HOLD: {shift_as_hold}")
        print(f"  - HOLD predicted as SHIFT: {hold_as_shift}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("="*80 + "\n")
    
    def _log_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: list,
        events: Dict,
        output_dir: str = "."
    ):
        """Log predictions to file sorted by time."""
        label_names = {0: "HOLD", 1: "SHIFT"}
        
        # Reconstruct event list with timestamps (same order as _get_shifts_holds)
        event_list = []
        
        # Process shifts first (same order as in _get_shifts_holds)
        for speaker, shifts in events.get('shifts', {}).items():
            for shift in shifts:
                event_list.append((shift, 1, int(speaker)))  # (time, label, speaker)
        
        # Then process holds (same order as in _get_shifts_holds)
        for speaker, holds in events.get('holds', {}).items():
            for hold in holds:
                event_list.append((hold, 0, int(speaker)))  # (time, label, speaker)
        
        # Combine with predictions
        predictions_data = []
        for i in range(len(y_true)):
            if i < len(event_list):
                time, true_label, speaker = event_list[i]
                predictions_data.append({
                    'time': time,
                    'speaker': speaker,
                    'true_label': true_label,
                    'pred_label': y_pred[i],
                    'score': scores[i]
                })
        
        # Sort by time
        predictions_data.sort(key=lambda x: x['time'])
        
        # Write to file in specified output directory
        from pathlib import Path
        output_path = Path(output_dir) / 'predictions.log'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"{'Time':<10} {'Speaker':<8} {'True':<8} {'Pred':<8} {'Score':<10} {'Match':<8}\n")
            f.write("="*60 + "\n")
            for item in predictions_data:
                match = "Yes" if item['true_label'] == item['pred_label'] else "No"
                f.write(
                    f"{item['time']:<10.2f} "
                    f"{item['speaker']:<8} "
                    f"{label_names[item['true_label']]:<8} "
                    f"{label_names[item['pred_label']]:<8} "
                    f"{item['score']:<10.3f} "
                    f"{match:<8}\n"
                )
        
        print(f"\nPredictions logged to {output_path} (sorted by time)\n")
    
    def _empty_results(self) -> Dict[str, float]:
        """Return empty results when no labels exist."""
        return {
            'shift_f1': 0.0,
            'hold_f1': 0.0,
            'weighted_f1': 0.0,
            'shift_precision': 0.0,
            'hold_precision': 0.0,
            'shift_recall': 0.0,
            'hold_recall': 0.0,
            'n_shifts': 0,
            'n_holds': 0
        }

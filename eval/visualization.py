"""
Visualization utilities for shift/hold predictions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Visualizer:
    """Handles visualization of predictions and ground truth."""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_detailed_comparison(
        self,
        p_future: torch.Tensor,
        vads: torch.Tensor,
        events: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        audio_path: str,
        feature_sr: int = 50,
        start_time: float = 0,
        stop_time: Optional[float] = None,
        output_name: str = "shift_hold_detailed.png"
    ):
        """
        Plot detailed comparison with audio, VAD, VAP probabilities, and labels.
        
        Args:
            p_future: Prediction tensor [T, 2]
            vads: VAD tensor [T, 2]
            events: Ground truth events dict
            y_true: Ground truth labels for events (1=shift, 0=hold)
            y_pred: Predicted labels for events (1=shift, 0=hold)
            audio_path: Path to audio file
            feature_sr: Feature sampling rate
            start_time: Start time in seconds
            stop_time: Stop time in seconds (None = full duration)
            output_name: Output filename
        """
        # Get probabilities
        p_shift = p_future[:, 0].cpu().numpy()
        p_hold = p_future[:, 1].cpu().numpy()
        
        # Get VAD for each speaker
        vad_0 = vads[:, 0].cpu().numpy()
        vad_1 = vads[:, 1].cpu().numpy()
        
        # Load audio
        sr, audio = wavfile.read(audio_path)
        audio = audio.astype('float')
        
        # Separate left and right channels
        if len(audio.shape) == 2:  # Stereo
            audio_left = audio[:, 0] / np.abs(audio[:, 0]).max()
            audio_right = audio[:, 1] / np.abs(audio[:, 1]).max()
        else:  # Mono
            audio_left = audio / np.abs(audio).max()
            audio_right = audio / np.abs(audio).max()
        
        # Create audio time axis
        audio_time = np.arange(len(audio_left)) / sr
        
        # Build event timeline from y_true and y_pred
        max_time = p_future.shape[0] / feature_sr
        
        # Reconstruct event list with timestamps (must match order in evaluator._get_shifts_holds)
        event_list = []
        # Process shifts first (same order as in evaluator)
        for speaker, shifts in events.get('shifts', {}).items():
            for shift in shifts:
                if shift <= max_time:
                    event_list.append((shift, 1, int(speaker)))  # (time, true_label, speaker)
        # Then process holds (same order as in evaluator)
        for speaker, holds in events.get('holds', {}).items():
            for hold in holds:
                if hold <= max_time:
                    event_list.append((hold, 0, int(speaker)))  # (time, true_label, speaker)
        # DO NOT sort - keep the same order to match y_true/y_pred indices
        
        # Build full timeline arrays for visualization
        gt = np.full(int(max_time * feature_sr), -1, dtype=int)
        predictions = np.full(int(max_time * feature_sr), -1, dtype=int)
        
        # Mark events in timeline
        for i, (event_time, _, _) in enumerate(event_list):
            if i < len(y_true):  # Make sure we have labels
                idx = int(event_time * feature_sr)
                if idx < len(gt):
                    gt[idx] = y_true[i]
                    predictions[idx] = y_pred[i]
        
        # Create time axis
        time_axis = np.arange(len(gt)) / feature_sr
        
        # Apply time window
        if stop_time is None:
            stop_time = max_time
        
        mask = (time_axis >= start_time) & (time_axis <= stop_time)
        time_window = time_axis[mask]
        pred_window = predictions[mask]
        gt_window = gt[mask]
        p_shift_window = p_shift[mask]
        p_hold_window = p_hold[mask]
        vad_0_window = vad_0[mask]
        vad_1_window = vad_1[mask]
        
        # Audio window mask
        audio_mask = (audio_time >= start_time) & (audio_time <= stop_time)
        audio_time_window = audio_time[audio_mask]
        audio_left_window = audio_left[audio_mask]
        audio_right_window = audio_right[audio_mask]
        
        # Create figure with 4 rows
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, height_ratios=[1.5, 1.5, 1, 1])
        
        fig.suptitle('Turn-Taking Analysis', fontsize=16, fontweight='bold')
        
        # ===== ROW 1: Audio + VAD for each channel =====
        self._plot_channel(
            fig, gs[0, 0], 
            audio_time_window, audio_left_window, 
            time_window, vad_0_window,
            start_time, stop_time,
            "Left Channel (Speaker 0)", 'r'
        )
        
        self._plot_channel(
            fig, gs[0, 1],
            audio_time_window, audio_right_window,
            time_window, vad_1_window,
            start_time, stop_time,
            "Right Channel (Speaker 1)", 'g'
        )
        
        # ===== ROW 2: VAP Probabilities =====
        ax_vap = fig.add_subplot(gs[1, :])
        ax_vap.plot(time_window, p_shift_window, color='#e74c3c', linewidth=2, label='P(Speaker 0)', alpha=0.8)
        ax_vap.plot(time_window, p_hold_window, color='#3498db', linewidth=2, label='P(Speaker 1)', alpha=0.8)
        ax_vap.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax_vap.set_ylim([0, 1])
        ax_vap.set_ylabel('Probability', fontsize=11)
        ax_vap.set_title('VAP Predictions', fontsize=12, fontweight='bold')
        ax_vap.legend(loc='upper right', fontsize=10)
        ax_vap.grid(True, alpha=0.3)
        ax_vap.set_xlim(start_time, stop_time)
        ax_vap.tick_params(labelbottom=False)
        
        # ===== ROW 3: Predicted Labels =====
        ax_pred = fig.add_subplot(gs[2, :])
        self._plot_shift_hold_trace(ax_pred, time_window, pred_window, "Predicted Labels")
        ax_pred.set_xlim(start_time, stop_time)
        ax_pred.tick_params(labelbottom=False)
        
        # ===== ROW 4: Ground Truth Labels =====
        ax_gt = fig.add_subplot(gs[3, :])
        self._plot_shift_hold_trace(ax_gt, time_window, gt_window, "Ground Truth Labels")
        ax_gt.set_xlim(start_time, stop_time)
        ax_gt.set_xlabel('Time (seconds)', fontsize=12)
        
        # Add legend
        shift_patch = mpatches.Patch(color='#e74c3c', label='Hold')
        hold_patch = mpatches.Patch(color='#3498db', label='Shift')
        unlabeled_patch = mpatches.Patch(color='#95a5a6', label='Unlabeled')
        fig.legend(handles=[shift_patch, hold_patch, unlabeled_patch], 
                  loc='lower center', fontsize=11, ncol=3, bbox_to_anchor=(0.5, -0.01))
        
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to: {output_path}")
    
    def _plot_channel(
        self, 
        fig, 
        gs_position, 
        audio_time, 
        audio_wave, 
        vad_time, 
        vad_signal,
        start_time, 
        stop_time, 
        title, 
        audio_color
    ):
        """Helper to plot audio waveform and VAD for one channel."""
        ax = fig.add_subplot(gs_position)
        
        # Plot audio waveform on secondary y-axis
        ax_audio = ax.twinx()
        ax_audio.plot(audio_time, audio_wave, audio_color, alpha=0.5, linewidth=0.5)
        ax_audio.set_ylim([-1, 1])
        ax_audio.set_ylabel('Audio Amplitude', fontsize=10, color=audio_color)
        ax_audio.tick_params(axis='y', labelcolor=audio_color)
        
        # Plot VAD on primary y-axis
        ax.plot(vad_time, vad_signal, 'k-', linewidth=2, label='VAD')
        ax.set_ylim([-0.1, 1.1])
        ax.set_ylabel('VAD', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(start_time, stop_time)
        ax.tick_params(labelbottom=False)
    
    def _plot_shift_hold_trace(self, ax, time_axis, labels, title):
        """Helper to plot shift/hold trace with color coding."""
        # Color mapping: -1=gray (unlabeled), 0=red (shift), 1=blue (hold)
        colors = {-1: '#95a5a6', 0: '#e74c3c', 1: '#3498db'}
        
        # Plot as step function
        for i in range(len(time_axis) - 1):
            label = labels[i]
            color = colors.get(label, '#95a5a6')
            ax.fill_between(
                [time_axis[i], time_axis[i+1]], 
                0, 1, 
                color=color, 
                alpha=0.7,
                step='post'
            )
        
        # Add event markers
        shift_times = time_axis[labels == 0]
        hold_times = time_axis[labels == 1]
        
        ax.scatter(shift_times, [0.5] * len(shift_times), 
                  color='#c0392b', s=50, marker='o', zorder=3, alpha=0.8)
        ax.scatter(hold_times, [0.5] * len(hold_times), 
                  color='#2980b9', s=50, marker='s', zorder=3, alpha=0.8)
        
        ax.set_ylabel('Event', fontsize=11)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['', 'Events', ''])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(-0.1, 1.1)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_name: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix for shift/hold predictions.
        
        Args:
            y_true: Ground truth labels (0=hold, 1=shift)
            y_pred: Predicted labels (0=hold, 1=shift)
            output_name: Output filename
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix as heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        classes = ['HOLD', 'SHIFT']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted Label',
               ylabel='True Label',
               title='Confusion Matrix: Shift/Hold Prediction')
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=20, fontweight='bold')
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i + 0.3, f'({cm_percent[i, j]:.1f}%)',
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12)
        
        # Add accuracy in the title
        accuracy = np.trace(cm) / np.sum(cm) * 100
        ax.set_title(f'Confusion Matrix: Shift/Hold Prediction\\nAccuracy: {accuracy:.2f}%', 
                    fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix to: {output_path}")

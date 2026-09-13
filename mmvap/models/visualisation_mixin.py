import matplotlib.pyplot as plt
import numpy as np


class VisualisationMixin:
    """
    Mixin providing attention visualisation and embedding extraction capabilities.

    Models should call self._init_viz_state() in their __init__ and then call
    self._maybe_accumulate_attention() / self._maybe_accumulate_embeddings() in
    their forward() pass.
    """

    def _init_viz_state(self):
        self.chunk_count = 0
        self.target_time_range = None
        self.window_size_sec = 10.0
        self.step_size_sec = 9.0
        self.accumulated_attentions = []
        self.visualization_complete = False
        self.output_name = None
        self.event_timestamps = []
        self.accumulated_embeddings = []
        self.embedding_window_sec = 1.0

    # ------------------------------------------------------------------
    # Public configuration / retrieval API
    # ------------------------------------------------------------------

    def set_visualization_time_range_and_name(
        self,
        start_time_sec: float,
        end_time_sec: float,
        output_name: str,
        window_size_sec: float = 10.0,
        step_size_sec: float = 9.0,
    ):
        self.target_time_range = (
            (start_time_sec, end_time_sec)
            if start_time_sec is not None and end_time_sec is not None
            else None
        )
        self.window_size_sec = window_size_sec
        self.step_size_sec = step_size_sec
        self.chunk_count = 0
        self.accumulated_attentions = []
        self.visualization_complete = False
        self.output_name = output_name

        if self.target_time_range is not None:
            print(f"Set visualization target range: [{start_time_sec:.2f}s - {end_time_sec:.2f}s]")
            print(f"  Window size: {window_size_sec}s, Step size: {step_size_sec}s")

            first_chunk = int(start_time_sec / step_size_sec)
            last_chunk = int((end_time_sec - 0.001) / step_size_sec)

            if first_chunk == last_chunk:
                print(f"  Range contained in chunk {first_chunk}")
            else:
                print(f"  Range spans chunks {first_chunk} to {last_chunk}")
                for chunk_idx in range(first_chunk, last_chunk + 1):
                    chunk_start = chunk_idx * step_size_sec
                    chunk_end = chunk_start + window_size_sec
                    print(f"    Chunk {chunk_idx}: [{chunk_start:.2f}s - {chunk_end:.2f}s]")

    def set_event_timestamps(self, event_timestamps: list, window_sec: float = 1.0):
        self.event_timestamps = event_timestamps
        self.embedding_window_sec = window_sec
        self.accumulated_embeddings = []
        self.chunk_count = 0

        if self.event_timestamps:
            print(f"Set {len(self.event_timestamps)} event timestamps for embedding extraction:")
            for event_type, timestamp in self.event_timestamps[:5]:
                print(f"  {event_type} at {timestamp:.2f}s")
            if len(self.event_timestamps) > 5:
                print(f"  ... and {len(self.event_timestamps) - 5} more")
            print(f"Extraction window: ±{window_sec}s around each event")

    def get_accumulated_embeddings(self):
        return self.accumulated_embeddings

    def reset_embedding_extraction(self):
        self.event_timestamps = []
        self.accumulated_embeddings = []
        self.chunk_count = 0

    # ------------------------------------------------------------------
    # Helpers called from forward()
    # ------------------------------------------------------------------

    def _maybe_accumulate_attention(self, cross_attn, chunk_start_time: float):
        """Accumulate cross-attention weights for the target time range."""
        if self.target_time_range is None or self.visualization_complete:
            return

        frames_per_sec = 50
        target_start, target_end = self.target_time_range
        chunk_end_time = chunk_start_time + self.window_size_sec

        if chunk_start_time >= target_end or chunk_end_time <= target_start:
            return

        overlap_start = max(target_start, chunk_start_time)
        overlap_end = min(target_end, chunk_end_time)

        overlap_frames = int(frames_per_sec * (self.window_size_sec - self.step_size_sec))
        frame_start = int((overlap_start - chunk_start_time) * frames_per_sec)
        frame_end = int((overlap_end - chunk_start_time) * frames_per_sec)
        frame_end = min(frame_end, cross_attn.shape[-1])

        if self.chunk_count > 1:
            frame_start = max(frame_start, overlap_frames)

        print(f'overlap frames: {overlap_frames}')
        attn_averaged = cross_attn.mean(dim=(2, 3))  # [B, 2, N, N]

        for frame_idx in range(frame_start, frame_end):
            absolute_time = chunk_start_time + frame_idx / frames_per_sec
            self.accumulated_attentions.append({
                'time': absolute_time,
                'chunk_start': chunk_start_time,
                'attn': attn_averaged[:, :, frame_idx, :].detach().cpu(),
            })

        frames_collected_str = f"{frame_start}-{frame_end}"
        if self.chunk_count > 1:
            frames_collected_str += f" (skipped overlap frames 0-{overlap_frames})"
        print(f"Collected frames {frames_collected_str} from chunk {self.chunk_count-1} [{chunk_start_time:.2f}s-{chunk_end_time:.2f}s]")
        print(f"  Overlap: [{overlap_start:.2f}s - {overlap_end:.2f}s]")

        if overlap_end >= target_end:
            print(f"\n{'='*60}")
            print(f"Completed collecting attention for range [{target_start:.2f}s - {target_end:.2f}s]")
            print(f"Total frames collected: {len(self.accumulated_attentions)}")
            print(f"{'='*60}\n")
            VisualisationMixin.visualize_attention_range(
                accumulated_data=self.accumulated_attentions,
                target_range=(target_start, target_end),
                window_size_sec=self.window_size_sec,
                output_path=self.output_name,
            )
            self.visualization_complete = True

    def _maybe_accumulate_embeddings(self, embeddings_tensor, chunk_start_time: float):
        """Accumulate hidden-state embeddings around configured event timestamps."""
        if not self.event_timestamps:
            return

        frames_per_sec = 50
        chunk_end_time = chunk_start_time + self.window_size_sec
        overlap_frames = int(frames_per_sec * (self.window_size_sec - self.step_size_sec))

        for event_type, event_time in self.event_timestamps:
            extract_start = event_time - self.embedding_window_sec
            extract_end = event_time + self.embedding_window_sec

            if chunk_start_time >= extract_end or chunk_end_time <= extract_start:
                continue

            overlap_start = max(extract_start, chunk_start_time)
            overlap_end = min(extract_end, chunk_end_time)

            frame_start = int((overlap_start - chunk_start_time) * frames_per_sec)
            frame_end = int((overlap_end - chunk_start_time) * frames_per_sec)
            frame_end = min(frame_end, embeddings_tensor.shape[1])

            if self.chunk_count > 1:
                frame_start = max(frame_start, overlap_frames)

            embeddings = embeddings_tensor[:, frame_start:frame_end, :].detach().cpu()

            for i, frame_idx in enumerate(range(frame_start, frame_end)):
                absolute_time = chunk_start_time + frame_idx / frames_per_sec
                self.accumulated_embeddings.append({
                    'event_type': event_type,
                    'event_time': event_time,
                    'absolute_time': absolute_time,
                    'relative_time': absolute_time - event_time,
                    'embedding': embeddings[:, i, :],
                    'chunk_idx': self.chunk_count - 1,
                })

    # ------------------------------------------------------------------
    # Static plotting method
    # ------------------------------------------------------------------

    @staticmethod
    def visualize_attention_range(
        accumulated_data: list,
        target_range: tuple,
        window_size_sec: float,
        output_path: str = 'attention_range.png',
    ):
        """
        Visualize attention weights across a time range (potentially spanning multiple chunks).

        Args:
            accumulated_data: List of dicts with 'time', 'chunk_start', 'attn' keys
            target_range: Tuple (start_sec, end_sec) of the target range
            window_size_sec: Window size in seconds
            output_path: Where to save the visualization
        """
        frames_per_sec = 50
        target_start, target_end = target_range

        accumulated_data = sorted(accumulated_data, key=lambda x: x['time'])

        times = np.array([d['time'] for d in accumulated_data])
        chunk_starts = np.array([d['chunk_start'] for d in accumulated_data])
        n_frames = len(accumulated_data)

        attn_ch0 = np.stack([d['attn'][0, 0].numpy() for d in accumulated_data])
        attn_ch1 = np.stack([d['attn'][0, 1].numpy() for d in accumulated_data])

        N = attn_ch0.shape[1]

        # # SANITY CHECK: Print raw attention for a few prediction points
        # with open('attn_ch0.json', 'w') as f:
        #     import json
        #     json.dump(attn_ch0.tolist(), f, indent=2)
        # print(f"\n=== SANITY CHECK ===")
        # if n_frames >= 3:
        #     for check_idx in [0, n_frames//2, n_frames-1]:
        #         pred_time = times[check_idx]
        #         chunk_start = chunk_starts[check_idx]
        #         attn_weights = attn_ch0[check_idx]
        #         max_attn_idx = np.argmax(attn_weights)
        #         max_attn_input_time = chunk_start + max_attn_idx / frames_per_sec
        #         print(f"Prediction time {pred_time:.3f}s (chunk {chunk_start:.1f}s):")
        #         print(f"  Max attention at input index {max_attn_idx}/{N} -> input time {max_attn_input_time:.3f}s")
        #         print(f"  Max attention value: {attn_weights[max_attn_idx]:.4f}")
        #         print(f"  Should be on diagonal? {abs(pred_time - max_attn_input_time) < 0.05}")
        # print(f"===================\n")
        # # END SANITY CHECK

        input_times_matrix = np.zeros((n_frames, N))
        for i, chunk_start in enumerate(chunk_starts):
            input_times_matrix[i, :] = chunk_start + np.arange(N) / frames_per_sec

        input_time_min = input_times_matrix.min()
        input_time_max = input_times_matrix.max()

        time_resolution = 1.0 / frames_per_sec
        common_input_times = np.arange(input_time_min, input_time_max + time_resolution, time_resolution)
        n_input_times = len(common_input_times)

        fig = plt.figure(figsize=(30, 10))
        gs = fig.add_gridspec(1, 42, hspace=0.2, wspace=0.4)

        fig.suptitle(
            f'Attention Across Time Range [{target_start:.2f}s - {target_end:.2f}s] ({n_frames} frames)',
            fontsize=18, fontweight='bold', y=0.96,
        )

        for channel_idx, attn_data in enumerate([attn_ch0, attn_ch1]):
            col_start = channel_idx * 21
            ax_heatmap = fig.add_subplot(gs[0, col_start:col_start + 19])

            attn_remapped = np.zeros((n_frames, n_input_times))
            for i in range(n_frames):
                chunk_start = chunk_starts[i]
                chunk_input_times = chunk_start + np.arange(N) / frames_per_sec
                indices = np.searchsorted(common_input_times, chunk_input_times, side='left')
                indices = np.clip(indices, 0, n_input_times - 1)
                attn_remapped[i, indices] = attn_data[i, :]

            pred_times_edges = np.concatenate([
                [times[0] - time_resolution / 2],
                (times[:-1] + times[1:]) / 2,
                [times[-1] + time_resolution / 2],
            ])
            input_times_edges = np.concatenate([
                [common_input_times[0] - time_resolution / 2],
                (common_input_times[:-1] + common_input_times[1:]) / 2,
                [common_input_times[-1] + time_resolution / 2],
            ])

            X, Y = np.meshgrid(pred_times_edges, input_times_edges)
            im = ax_heatmap.pcolormesh(X, Y, attn_remapped.T, cmap='hot', shading='flat')

            ax_heatmap.set_aspect('equal')
            ax_heatmap.set_xlim(target_start, target_end)
            ax_heatmap.set_ylim(target_start, target_end)

            ax_heatmap.plot(
                [target_start, target_end], [target_start, target_end],
                'cyan', linewidth=2, alpha=0.7, label='Current time (diagonal)', zorder=10,
            )
            ax_heatmap.axvline(x=times[0], color='lime', linestyle=':', linewidth=1, alpha=0.5, zorder=5)
            ax_heatmap.axvline(x=times[-1], color='lime', linestyle=':', linewidth=1, alpha=0.5, zorder=5)
            ax_heatmap.axhline(y=input_time_min, color='orange', linestyle=':', linewidth=1, alpha=0.5, zorder=5)
            ax_heatmap.axhline(y=input_time_max, color='orange', linestyle=':', linewidth=1, alpha=0.5, zorder=5)

            ax_heatmap.set_xlabel('Prediction Time (seconds, absolute)', fontsize=13)
            ax_heatmap.set_ylabel('Input Time (seconds, absolute)', fontsize=13)
            ax_heatmap.set_title(
                f'Channel {channel_idx} - Attention Heatmap\nShows which input times the model attends to for each prediction',
                fontsize=12, fontweight='bold', pad=10,
            )
            ax_heatmap.legend(loc='upper left', fontsize=10)
            ax_heatmap.grid(True, alpha=0.2, linestyle='--')

            ax_cbar = fig.add_subplot(gs[0, col_start + 20])
            cbar = plt.colorbar(im, cax=ax_cbar)
            cbar.set_label('Attention Weight', fontsize=11)

            max_val = attn_data.max()
            mean_val = attn_data.mean()
            std_val = attn_data.std()
            textstr = f'Max: {max_val:.4f}\nMean: {mean_val:.4f}\nStd: {std_val:.4f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
            ax_heatmap.text(
                0.98, 0.02, textstr, transform=ax_heatmap.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=props,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved attention range visualization to {output_path}')
        print(f'  Time range: [{target_start:.2f}s - {target_end:.2f}s]')
        print(f'  Frames: {n_frames}')
        print(f'  Channel 0: max={attn_ch0.max():.4f}, mean={attn_ch0.mean():.4f}')
        print(f'  Channel 1: max={attn_ch1.max():.4f}, mean={attn_ch1.mean():.4f}')

"""
Model inference utilities for VAP model.
"""

import yaml
import torch
from typing import Tuple, Dict

from mmvap.models.multimodal_model import EarlyVAFusion
from mmvap.models.model import StereoTransformerModel, StereoTransformerModelVideoOnly
from mmvap.data.dataloader import (
    AVCocktailValidationAudioVisualDataset,
    ValidationAudioVisualDataset,
    ValidationAudioDataset,
)
from mmvap.data.codebook import VAPDecoder


def load_model(model_name: str, config_path: str, weights_path: str, device: str = "cuda:0") -> EarlyVAFusion:
    """
    Load VAP model from config and weights.
    
    Args:
        config_path: Path to model config YAML
        weights_path: Path to model weights
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model '{model_name}' from:")
    print(f"  Config: {config_path}")
    print(f"  Weights: {weights_path}")

    cfg = yaml.safe_load(open(config_path, "r"))
    if model_name == "early_fusion_candor" or model_name.startswith("early_fusion"):
        model = EarlyVAFusion(cfg=cfg)
    elif model_name.startswith("vap"):
        model = StereoTransformerModel(cfg=cfg)
    elif model_name.startswith("video"):
        model = StereoTransformerModelVideoOnly(cfg=cfg)
    elif model_name == "late_fusion":
        from mmvap.models.multimodal_model import LateVAFusion
        model = LateVAFusion(cfg=cfg)

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)["state_dict"]
        # replace "model." prefix in state_dict keys if it exists (Lightning wrapper)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
    except KeyError:
        # If "state_dict" key doesn't exist, assume the entire dict is the state dict
        new_state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    # print(f"New state dict keys: {list(new_state_dict.keys())}")
    model.load_state_dict(new_state_dict)
    
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model: EarlyVAFusion,
    audio_path: str,
    video_dir: str,
    channelmap: str,
    transcript_path: str = None,
    window_size: int = 20,
    step_size: int = 19,
    mode: str = "VAP",
    feature_sr: int = 50,
    data: str = "avcocktail",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run model inference on audio-visual input.
    
    Args:
        model: VAP model
        audio_path: Path to audio file
        video_dir: Directory containing video features
        channelmap: Path to channel mapping file
        transcript_path: Path to transcript file (optional)
        window_size: Window size for inference
        step_size: Step size for inference
        mode: Inference mode
        feature_sr: Feature sampling rate
    
    Returns:
        vaps: VAP predictions [T, 256]
        vads: VAD predictions [T, 2]
    """
    if data == "avcocktail":
        dataset = AVCocktailValidationAudioVisualDataset(
            video_pkl_dir=video_dir,
            channelmap=channelmap,
            audio_file=audio_path,
            transcript_file=transcript_path,
            sr=16_000,
            feature_sr=feature_sr,
            window_size=window_size,
            step_size=step_size,
            mode=mode
        )
    elif data == "candor":
        dataset = ValidationAudioVisualDataset(
            video_pkl_dir=video_dir,
            channelmap=channelmap,
            audio_file=audio_path,
            transcript_file=transcript_path,
            sr=16_000,
            feature_sr=feature_sr,
            window_size=window_size,
            step_size=step_size,
            mode=mode
        )
    elif data == "switchboard":
        dataset = ValidationAudioDataset(
            audio_file=audio_path,
            transcript_file=transcript_path,
            sr=16_000,
            feature_sr=feature_sr,
            window_size=window_size,
            step_size=step_size,
            mode=mode
        )
    overlap = int(feature_sr * (window_size - step_size))

    vaps_list = []
    vads_list = []

    for idx in range(len(dataset)):
        sample = dataset[idx]

        batch = {'audio_chunk': sample['audio_chunk'].unsqueeze(0)}
        if 'frames' in sample:
            batch['frames'] = sample['frames'].unsqueeze(0)

        with torch.no_grad():
            vad_pred, vap_pred = model(batch)

        if mode == 'VAP':
            vap_pred = torch.softmax(vap_pred, dim=-1).cpu()
        else:
            vap_pred = torch.sigmoid(vap_pred).cpu()
        vad_pred = torch.sigmoid(vad_pred).cpu()

        if idx == 0:
            vaps_list.append(vap_pred[0, :overlap, ...])
            vads_list.append(vad_pred[0, :overlap, :])

        vaps_list.append(vap_pred[0, overlap:, ...])
        vads_list.append(vad_pred[0, overlap:, :])

    vaps = torch.cat(vaps_list, dim=0)
    vads = torch.cat(vads_list, dim=0)
    return vaps, vads


def decode_predictions(vaps: torch.Tensor, mode: str = "VAP") -> Dict[str, torch.Tensor]:
    """
    Decode VAP predictions into probabilities.

    Args:
        vaps: Raw predictions
            - VAP mode: [T, 256] or [B, T, 256] discrete states
            - ind-4 mode: [T, 8] or [B, T, 8] where 8 = 2 speakers × 4 time bins
            - ind-40 mode: [T, 80] or [B, T, 80] where 80 = 2 speakers × 40 time bins
        mode: Prediction mode ('VAP', 'ind-4', or 'ind-40')

    Returns:
        Dict with decoded probabilities:
            - p_future: Future predictions [T, 2] (bins 2-3 for VAP, bins 2-3 for ind-4)
            - p_now: Current predictions [T, 2]
            - p_bc: Backchannel predictions [T, 2]
            - p_all: All probabilities
    """
    # Remove batch dimension if present
    if len(vaps.shape) == 3:
        vaps = vaps.squeeze(0)

    if mode == 'VAP':
        # Use codebook decoder for discrete VAP predictions
        decoder = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])
        return {
            'p_future': decoder.p_future(vaps),
            'p_now': decoder.p_now(vaps),
            'p_bc': decoder.p_bc(vaps),
            'p_all': decoder.decode_probabilities(vaps)
        }
    elif mode in ['ind-4', 'ind-40']:
        # For independent prediction modes, vaps is already probabilities after sigmoid
        # Shape: [T, 2*num_bins] where first half is speaker 0, second half is speaker 1
        # For ind-4: [T, 8] = [T, 2 speakers × 4 bins]
        # For ind-40: [T, 80] = [T, 2 speakers × 40 bins]

        num_bins = 4 if mode == 'ind-4' else 40

        # Reshape to [T, 2, num_bins] for easier indexing
        vaps_reshaped = vaps.reshape(vaps.shape[0], 2, num_bins)

        # Extract different time horizons
        # p_now: bins 0-1 (0.0-0.4s)
        # p_future: bins 2-3 (0.4-0.8s)
        p_now_ch0 = vaps_reshaped[:, 0, 0:2].sum(dim=1)  # Sum bins 0-1 for speaker 0
        p_now_ch1 = vaps_reshaped[:, 1, 0:2].sum(dim=1)  # Sum bins 0-1 for speaker 1
        p_now = torch.stack([p_now_ch0, p_now_ch1], dim=1)  # [T, 2]

        p_future_ch0 = vaps_reshaped[:, 0, 2:4].sum(dim=1)  # Sum bins 2-3 for speaker 0
        p_future_ch1 = vaps_reshaped[:, 1, 2:4].sum(dim=1)  # Sum bins 2-3 for speaker 1
        p_future = torch.stack([p_future_ch0, p_future_ch1], dim=1)  # [T, 2]

        p_future /= p_future.sum(-1, keepdim=True) + 1e-5
        p_now /= p_now.sum(-1, keepdim=True) + 1e-5
        vaps_reshaped /= vaps_reshaped.sum(-1, keepdim=True) + 1e-5

        # For p_bc and p_all, we can return the full reshaped tensor
        return {
            'p_future': p_future,
            'p_now': p_now,
            'p_bc': p_future,  # Use same as p_future for now
            'p_all': vaps_reshaped  # [T, 2, num_bins]
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'VAP', 'ind-4', or 'ind-40'")


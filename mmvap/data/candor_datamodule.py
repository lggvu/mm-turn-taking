"""
PyTorch Lightning DataModule for Candor dataset.
Uses the AudioVisualDataset implementation.
"""

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import torch
from tqdm import tqdm
import pickle
import hashlib

from mmvap.data.dataloader import AudioVisualDataset
from mmvap.data.augmentation import FlipChannel, MaskVad


def collate_fn(batch_list):
    """
    Collate function to convert tuple format from AudioVisualDataset to dict format.
    
    AudioVisualDataset returns:
        (id, start, end, vad, vap, inverse_vap, audio_chunk, ch_l, ch_r, frames)
    
    Models expect dict format:
        {"id": ..., "start_time": ..., "vad": ..., "vap": ..., "audio_chunk": ..., "frames": ..., "channel_ids": [...]}
    """
    # Unpack tuples from each sample
    ids, starts, ends, vads, vaps, inverse_vaps, audio_chunks, ch_ls, ch_rs, frames_list = zip(*batch_list)
    
    # Stack tensors
    batch = {
        "id": ids,
        "start_time": starts,
        "end_time": ends,
        "vad": torch.stack(vads),
        "vap": torch.stack(vaps),
        "inverse_vap": torch.stack(inverse_vaps),
        "audio_chunk": torch.stack(audio_chunks),
        "channel_ids": list(zip(ch_ls, ch_rs)),
        "frames": torch.stack(frames_list),
    }
    
    return batch


class CandorDataModule(pl.LightningDataModule):
    """
    DataModule for Candor dataset with train/val splits.
    Uses the AudioVisualDataset implementation.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        # Training parameters from config
        self.batch_size = cfg['batch_size']
        self.num_workers = cfg['num_workers']
        self.pin_memory = cfg.get('pin_memory', False)
        
        # Augmentation parameters
        self.flip_channel_prob = cfg['flip_channel_prob']
        self.mask_vad_prob = cfg['mask_vad_prob']
        self.mask_vad_scale = cfg['mask_vad_scale']
        
        # Paths
        self.train_video_dir = cfg['train']['video_directory']
        self.train_channelmaps = cfg['train']['channelmaps']
        self.train_wavdir = cfg['train']['wavdir']
        self.train_pickle_base = cfg['train']['pickle_base']
        self.train_list = cfg['train'].get('train_list')
        
        self.val_video_dir = cfg['val']['video_directory']
        self.val_channelmaps = cfg['val']['channelmaps']
        self.val_wavdir = cfg['val']['wavdir']
        self.val_pickle_base = cfg['val']['pickle_base']
        self.val_list = cfg['val'].get('val_list')
        
        # Mode and fold
        self.mode = cfg['mode']
        self.n_folds = cfg.get('n_folds', 1)
        self.current_fold = cfg.get('current_fold', 0)
        self.fps = cfg.get('fps', 30)

        self.sr = cfg["sampling_rate"]

    def _get_cache_path(self, pickle_file: str, filter_list: Optional[str], split: str, sr: int) -> str:
        """Generate cache path based on dataset configuration."""
        # Create hash from configuration to ensure cache invalidation on config changes
        if sr != 16000:
            config_str = f"{pickle_file}|{filter_list}|{split}|{self.mode}|{self.fps}|{sr}"
        else:
            config_str = f"{pickle_file}|{filter_list}|{split}|{self.mode}|{self.fps}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'filtered_indices_{config_hash}.pkl')

    def _load_or_create_filtered_dataset(self, dataset, filter_list: Optional[str], 
                                          pickle_file: str, split: str, sr: int):
        """Load cached filtered indices or create new ones."""
        if filter_list is None:
            return dataset
        
        cache_path = self._get_cache_path(pickle_file, filter_list, split, sr)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"Loading cached {split} indices from {cache_path}")
            with open(cache_path, 'rb') as f:
                filtered_ids = pickle.load(f)
            print(f"Loaded {len(filtered_ids)} cached {split} samples")
            return torch.utils.data.Subset(dataset, filtered_ids)
        
        # Create filtered indices
        print(f"Creating filtered {split} indices (this will be cached for future runs)")
        import pandas as pd
        df = pd.read_csv(filter_list)
        filtered_ids = [
            i for i, sample in tqdm(enumerate(dataset), desc=f"Filtering {split} set", total=len(dataset))
            if sample[0] in df['id'].values
        ]
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(filtered_ids, f)
        print(f"Cached {len(filtered_ids)} {split} samples to {cache_path}")
        
        return torch.utils.data.Subset(dataset, filtered_ids)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        fold_name = f"fold_{self.current_fold}_30s"
        
        if stage == "fit" or stage is None:
            # Training dataset
            train_pickle = os.path.join(self.train_pickle_base, fold_name, "train")
            train_dataset_full = AudioVisualDataset(
                video_directory=self.train_video_dir,
                video_format='.pkl',
                channelmaps=self.train_channelmaps,
                wavdir=self.train_wavdir,
                pickle_file=train_pickle,
                mode=self.mode,
                fps=self.fps,
                sr=self.sr
            )
            
            # Filter training set if list provided (with caching)
            self.train_dataset = self._load_or_create_filtered_dataset(
                train_dataset_full, self.train_list, train_pickle, "train", self.sr
            )
            
            # Validation dataset
            val_pickle = os.path.join(self.val_pickle_base, fold_name, "dev")
            val_dataset_full = AudioVisualDataset(
                video_directory=self.val_video_dir,
                video_format='.pkl',
                channelmaps=self.val_channelmaps,
                wavdir=self.val_wavdir,
                pickle_file=val_pickle,
                mode=self.mode,
                fps=self.fps,
                sr=self.sr
            )
            
            # Filter validation set if list provided (with caching)
            self.val_dataset = self._load_or_create_filtered_dataset(
                val_dataset_full, self.val_list, val_pickle, "val", self.sr
            )
            
            # Setup augmentation callbacks
            self.flip_channel = FlipChannel(probability=self.flip_channel_prob)
            self.mask_vad = MaskVad(
                probability=self.mask_vad_prob,
                feature_hz=50,
                audio_hz=16000,
                scale=self.mask_vad_scale,
            )

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1,
            collate_fn=collate_fn,
        )

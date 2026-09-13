"""
PyTorch Lightning DataModule for AVCocktail dataset.
Uses the original dataset implementation.
"""

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import torch
from tqdm import tqdm

from mmvap.data.dataset import AVCocktailDataset
from mmvap.data.augmentation import FlipChannel, MaskVad


def collate_fn(batch_list):
    """
    Collate function to convert tuple format from AVCocktailDataset to dict format.
    
    AVCocktailDataset returns:
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
    # print(batch)
    return batch


class AVCocktailDataModule(pl.LightningDataModule):
    """
    DataModule for AVCocktail dataset with train/val splits.
    Uses the original AVCocktailDataset implementation.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        # Training parameters from config or parent cfg
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
        self.train_list = cfg['train']['train_list']
        
        self.val_video_dir = cfg['val']['video_directory']
        self.val_channelmaps = cfg['val']['channelmaps']
        self.val_wavdir = cfg['val']['wavdir']
        self.val_pickle_base = cfg['val']['pickle_base']
        self.val_list = cfg['val']['val_list']
        # print(self.train_list, self.val_list)
        
        
        # Mode and fold
        self.mode = cfg['mode']
        # print(f"DataModule mode: {self.mode}")
        self.n_folds = cfg.get('n_folds', 1)
        self.current_fold = cfg.get('current_fold', 0)

        self.sr = cfg['sampling_rate']

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        fold_name = f"fold_{self.current_fold}"
        
        if stage == "fit" or stage is None:
            # Training dataset
            train_pickle = os.path.join(self.train_pickle_base, fold_name, "train")
            self.train_dataset = AVCocktailDataset(
                video_directory=self.train_video_dir,
                channelmaps=self.train_channelmaps,
                wavdir=self.train_wavdir,
                pickle_file=train_pickle,
                dev=False,
                mode=self.mode,
                fps=self.cfg['fps'],
                sr=self.sr    
            )
            # print(self.train_dataset[0][0]) -> session id
            # Filter training set if list provided
            if self.train_list is not None:
                import pandas as pd
                train_df = pd.read_csv(self.train_list)
                filtered_ids = [
                    i for i, sample in tqdm(enumerate(self.train_dataset), desc="Filtering training set", total=len(self.train_dataset))
                    if sample[0] in train_df['id'].values
                ]
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, filtered_ids)
                print(f"Filtered training set: {len(filtered_ids)} samples")
            
            # Validation dataset
            val_pickle = os.path.join(self.val_pickle_base, fold_name, "dev")
            self.val_dataset = AVCocktailDataset(
                video_directory=self.val_video_dir,
                channelmaps=self.val_channelmaps,
                wavdir=self.val_wavdir,
                pickle_file=val_pickle,
                dev=True,
                mode=self.mode,
                fps=self.cfg['fps'],
                sr=self.sr
            )
            
            if self.val_list is not None:
                import pandas as pd
                val_df = pd.read_csv(self.val_list)
                filtered_ids = [
                    i for i, sample in tqdm(enumerate(self.val_dataset), desc="Filtering validation set", total=len(self.val_dataset))
                    if sample[0] in val_df['id'].values
                ]
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, filtered_ids)
            # self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(0, 10)) # 10 samples

            
            # Setup augmentation callbacks
            self.flip_channel = FlipChannel(probability=self.flip_channel_prob)
            self.mask_vad = MaskVad(
                probability=self.mask_vad_prob,
                feature_hz=self.cfg['feature_hz'],
                audio_hz=self.sr,
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
            collate_fn=collate_fn,  # Convert tuple to dict
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=self.pin_memory,
            prefetch_factor=1,
            collate_fn=collate_fn,  # Convert tuple to dict
        )

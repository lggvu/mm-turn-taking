"""
PyTorch Lightning module for MM-VAP models.
Wraps the original PyTorch model implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import einops
from typing import Dict, Any, Optional

from mmvap.models.vap_models import multimodal_models, models


class  MMVAPLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for MM-VAP models.
    Uses the original model implementations from the old codebase.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        # self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Build model using old codebase
        self.model_name = cfg['model']['model_name']
        self.model = self._build_model(self.model_name, cfg)

        # Loss mode
        self.mode = cfg['model']['base']['mode'] # VAP, ind-40, ind-4

    def _build_model(self, model_name: str, cfg: Dict) -> nn.Module:
        """Build the model using the old codebase."""
        # The old codebase expects cfg in a specific format
        old_cfg = self._convert_config_format(cfg)
        self.save_hyperparameters(old_cfg)  # Save the converted config for logging
        print(f'multimodal_models: {multimodal_models}')
        # Get model class from old codebase
        if model_name in multimodal_models.__all__:
            model_class = getattr(multimodal_models, model_name)
        elif model_name in models.__all__:
            model_class = getattr(models, model_name)
        # model_class = getattr(multimodal_models, model_name)
        model = model_class(old_cfg)
    
        return model

    def _convert_config_format(self, cfg: Dict) -> Dict:
        """
        Convert new config format to old codebase format.
        The old models expect specific keys like 'multimodal_model_cfg', 'model_cfg', etc.
        """
        # Keep the multimodal config if it exists
        old_cfg = {}
        
        if 'multimodal' in cfg['model']:
            old_cfg['multimodal_model_cfg'] = cfg['model']['multimodal']
        
        if 'base' in cfg['model']:
            old_cfg['model_cfg'] = cfg['model']['base']
            # Also copy mode to top level for compatibility
            if 'mode' in cfg['model']['base']:
                old_cfg['model_cfg']['mode'] = cfg['model']['base']['mode']
        
        # Add encoder config if present - NEW: support flexible encoder configuration
        if 'encoder' in cfg:
            # Hydra merges the encoder/*.yaml into cfg['encoder']
            encoder_config = cfg['encoder']
            
            # Build encoder_cfg for the old model format
            old_cfg['encoder_cfg'] = {
                'class': encoder_config.get('class', 'StereoEncoderXLSR'),
                'params': {}
            }
            
            # Add encoder-specific parameters (CPC is the only encoder shipped in this release)
            if encoder_config.get('class') == 'StereoEncoder':
                old_cfg['encoder_cfg']['params'] = {}  # CPC has no special params
            # Also keep the old encoder format for backward compatibility
            if 'encoder' in cfg['model']:
                old_cfg['encoder_info'] = {
                    'audio_sample_rate': cfg['model']['encoder']['audio_sample_rate'],
                    'sample_rate': cfg['model']['encoder']['feature_sample_rate'],
                }
        
        # Add device
        old_cfg['device'] = cfg['trainer']['devices']
        
        # Add feature set
        if 'feature_set' in cfg['model']:
            old_cfg['feature_set'] = cfg['model']['feature_set']
        
        # Add training config for compatibility (even though we're not using it in Lightning)
        old_cfg['training_cfg'] = {
            'batch_size': cfg['training']['batch_size'],
            'flip_channel': cfg['augmentation']['flip_channel_prob'],
            'gradient_clip_val': cfg['trainer']['gradient_clip_val'],
            'learning_rate': cfg['optimizer']['lr'],
            'lr_scheduler': cfg['scheduler']['name'] if cfg['scheduler']['enabled'] else None,
            'mask_vad': cfg['augmentation']['mask_vad_prob'],
            'mask_vad_scale': cfg['augmentation']['mask_vad_scale'],
            'n_epochs': cfg['trainer']['max_epochs'],
        }
        
        return old_cfg

    def forward(self, batch: Dict) -> tuple:
        """Forward pass - delegates to the original model."""
        return self.model(batch)

    def _head_losses(self, vad_true, vad_pred, vap_true, vap_pred):
        """VAP/VAD loss for one prediction head. Single source of truth for the
        objective, shared by compute_loss (one head) and compute_layerwise_loss
        (looped once per probe layer)."""
        if self.mode == 'ind-40' or self.mode == 'ind-4':
            # Independent prediction mode
            vap_loss = F.binary_cross_entropy_with_logits(vap_pred, vap_true)
            vad_loss = F.binary_cross_entropy_with_logits(vad_pred, vad_true)
        else:
            # VAP mode (cross-entropy for vap, BCE for vad)
            vap_pred = einops.rearrange(vap_pred, "b n d -> (b n) d")
            vap_true = einops.rearrange(vap_true, "b n -> (b n)")
            vap_loss = F.cross_entropy(vap_pred, vap_true)
            vad_loss = F.binary_cross_entropy_with_logits(vad_pred, vad_true)
        return vap_loss, vad_loss

    def compute_loss(self, vad_true, vad_pred, vap_true, vap_pred):
        """
        Compute VAD and VAP losses.
        Uses the same loss function as the old codebase.
        """
        vap_loss, vad_loss = self._head_losses(vad_true, vad_pred, vap_true, vap_pred)
        total_loss = vap_loss + vad_loss
        return total_loss, vap_loss, vad_loss

    def compute_layerwise_loss(self, vad_true, vad_pred, vap_true, vap_pred):
        """
        vad_pred: [B, T, 2, L]; vap_pred: [B, T, D, L] -- one independent VAD/VAP head
        per AV-HuBERT layer (see StereoLinearProbeModel).

        total_loss is the SUM over layers, not the mean: the heads are
        parameter-disjoint, so summing gives each head exactly the gradient it would
        get if trained alone, whereas averaging would silently divide every head's
        gradient by L (an implicit lr/L).
        """
        vap_losses, vad_losses = [], []
        for l in range(vap_pred.shape[-1]):
            vap_l, vad_l = self._head_losses(vad_true, vad_pred[..., l], vap_true, vap_pred[..., l])
            vap_losses.append(vap_l)
            vad_losses.append(vad_l)
        total_loss = torch.stack(vap_losses).sum() + torch.stack(vad_losses).sum()
        return total_loss, vap_losses, vad_losses

    def _layerwise_accuracy(self, vad_true, vad_pred, vap_true, vap_pred):
        """Per-layer accuracy, each a python float list of length L. Cross-entropy
        loss is a less directly interpretable "performance" axis than accuracy for a
        layer-vs-performance plot."""
        vad_acc, vap_acc = [], []
        with torch.no_grad():
            for l in range(vap_pred.shape[-1]):
                vad_acc.append(((vad_pred[..., l] > 0).float() == vad_true).float().mean())
                if self.mode in ('ind-40', 'ind-4'):
                    vap_acc.append(((vap_pred[..., l] > 0).float() == vap_true).float().mean())
                else:
                    vap_acc.append((vap_pred[..., l].argmax(dim=-1) == vap_true).float().mean())
        return vad_acc, vap_acc

    def _log_layerwise(self, split, loss, vap_losses, vad_losses, batch_size, extra=None):
        """Shared logging for training_step/validation_step's layer-wise probe
        branch: the aggregate {split}/loss (kept so ModelCheckpoint(monitor='val/loss')
        still works, selecting on layer-average quality) plus one scalar per layer per
        metric. Per-layer scalars are epoch-only to avoid bloating the event file with
        L extra points every logged step."""
        n = len(vap_losses)
        self.log(f'{split}/loss', loss / batch_size, on_step=(split == 'train'), on_epoch=True, prog_bar=True)
        self.log(f'{split}/loss_mean_layer', loss / batch_size / n, on_step=(split == 'train'), on_epoch=True)

        per_layer = {}
        for l in range(n):
            per_layer[f'{split}/vap_loss_layer_{l:02d}'] = vap_losses[l] / batch_size
            per_layer[f'{split}/vad_loss_layer_{l:02d}'] = vad_losses[l] / batch_size
        if extra:
            per_layer.update(extra)
        self.log_dict(per_layer, on_step=False, on_epoch=True)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step - applies augmentation and computes loss."""
        # Apply augmentation (if datamodule has it)
        # print(f'self.trainer.datamodule: {self.trainer.datamodule}')
        if hasattr(self.trainer.datamodule, 'flip_channel'):
            batch = self.trainer.datamodule.flip_channel(batch)
        if hasattr(self.trainer.datamodule, 'mask_vad'):
            batch = self.trainer.datamodule.mask_vad(batch)
        
        # random modality dropout
        if hasattr(self.trainer.datamodule, 'modality_dropout'):
            batch = self.trainer.datamodule.modality_dropout(batch)

        # Forward pass
        vad_pred, vap_pred = self(batch)

        # Compute loss
        vad_true = batch['vad'].cuda()
        vap_true = batch['vap'].cuda()
        # print(f'train vad_true shape: {vad_true.shape}, vad_pred shape: {vad_pred.shape}')
        # print(f'vap_true shape: {vap_true.shape}, vap_pred shape: {vap_pred.shape}')
        # vad_true shape: torch.Size([4, 1000, 2]), vad_pred shape: torch.Size([4, 1000, 2])
        # vap_true shape: torch.Size([4, 1000]), vap_pred shape: torch.Size([4, 1000, 256])
        batch_size = self.cfg['training']['batch_size']

        loss, vap_loss, vad_loss = self.compute_loss(vad_true, vad_pred, vap_true, vap_pred)

        # Log metrics
        self.log('train/loss', loss / batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/vap_loss', vap_loss / batch_size, on_step=True, on_epoch=True)
        self.log('train/vad_loss', vad_loss / batch_size, on_step=True, on_epoch=True)

        return loss / batch_size

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step - no augmentation."""
        # Forward pass
        vad_pred, vap_pred = self(batch)

        # Compute loss
        vad_true = batch['vad'].cuda()
        vap_true = batch['vap'].cuda()
        # print(f'val vad_true shape: {vad_true.shape}, vad_pred shape: {vad_pred.shape}')
        batch_size = self.cfg['validation']['batch_size']

        loss, vap_loss, vad_loss = self.compute_loss(vad_true, vad_pred, vap_true, vap_pred)

        # Log metrics
        # print(f'loss: {loss}')
        # print(f'batch_size: {batch_size}')
        self.log('val/loss', loss / batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/vap_loss', vap_loss / batch_size, on_step=False, on_epoch=True)
        self.log('val/vad_loss', vad_loss / batch_size, on_step=False, on_epoch=True)

        return loss / batch_size


    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        cfg = self.cfg # this is the correct, new-format

        # Optimizer
        optimizer_cfg = cfg['optimizer']
        optimizer_name = optimizer_cfg['name']
        lr = optimizer_cfg['lr']
        betas = optimizer_cfg.get('betas')
        weight_decay = optimizer_cfg.get('weight_decay')
        adapter_cfg = optimizer_cfg.get('encoder_adapter') or {}

        if optimizer_name == 'AdamW':
            if adapter_cfg.get('enabled', False):
                param_groups = self._param_groups_with_adapter_split(lr, weight_decay, adapter_cfg)
                optimizer = torch.optim.AdamW(param_groups, betas=betas)
            else:
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=lr,
                    betas=betas,
                    weight_decay=weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler (optional)
        scheduler_cfg = cfg.get('scheduler', {})
        if scheduler_cfg.get('enabled', False):
            scheduler_name = scheduler_cfg.get('name')
            
            if scheduler_name == 'ReduceLROnPlateau':
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode=scheduler_cfg.get('mode', 'min'),
                    factor=scheduler_cfg.get('factor', 0.5),
                    patience=scheduler_cfg.get('patience', 2)
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val/loss',
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
            elif scheduler_name == 'LinearLR':
                from torch.optim.lr_scheduler import LinearLR
                scheduler = LinearLR(
                    optimizer,
                    start_factor=cfg.get('scheduler').get('start_factor'),
                    end_factor=cfg.get('scheduler').get('end_factor'),
                    total_iters=cfg.get('scheduler').get('total_iters'),
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1
                    }
                }
            else:
                raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return optimizer


"""
Training script for MM-VAP models using PyTorch Lightning and Hydra.
Usage:
    python scripts/train.py
    python scripts/train.py model=early_va_fusion data=avcocktail
    python scripts/train.py trainer.max_epochs=20 training.batch_size=8
    
    # Load from pretrained checkpoint:
    python scripts/train.py paths.pretrained_checkpoint=path/to/checkpoint.pt
    python scripts/train.py paths.pretrained_checkpoint=checkpoints/last.ckpt
"""

from datetime import datetime
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from mmvap.models import MMVAPLightningModule
from mmvap import data as mmvap_data


torch.use_deterministic_algorithms(True, warn_only=True)
def load_pretrained_weights(model: MMVAPLightningModule, checkpoint_path: str):
    """
    Load pre-trained weights from PyTorch or Lightning checkpoint.
    
    Args:
        model: The Lightning module to load weights into
        checkpoint_path: Path to checkpoint file (.pt or .ckpt)
    """
    print(f"Loading pretrained weights from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Determine checkpoint type and extract state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # => lightning checkpoint
        # Lightning checkpoint format
        print("Detected Lightning checkpoint format")
        state_dict = checkpoint['state_dict']
        
        # Check if weights are already prefixed with 'model.'
        # Lightning checkpoints should already have this prefix
        has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
        
        if has_model_prefix:
            # Direct load - keys should match
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        else:
            # Need to add 'model.' prefix
            new_state_dict = {f'model.{k}': v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    else:
        # Raw PyTorch state dict -> used to fine-tune externally-provided pretrained weights
        print("Detected PyTorch checkpoint format")
        
        # PyTorch checkpoints need to be loaded into model.model (the wrapped model)
        # We need to add the 'model.' prefix for Lightning
        new_state_dict = {f'model.{k}': v for k, v in checkpoint.items()}
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    # Report loading status
    if missing:
        print(f"Warning: Missing keys in checkpoint: {missing[:5]}..." if len(missing) > 5 else f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected[:5]}..." if len(unexpected) > 5 else f"Warning: Unexpected keys: {unexpected}")
    
    if not missing and not unexpected:
        print("Successfully loaded all weights")
    
    return model


@hydra.main(config_path="../configs", config_name="train_config")
def main(cfg: DictConfig):
    """Main training function."""
    
    print("="*80)
    print("Turn-taking Prediction Lightning Training")
    print("="*80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Set seed
    seed = cfg.training.get('seed')
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    
    print(f"cfg.optimizer: {cfg.optimizer}")
    
    # Merge data config with training config
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    data_cfg.update({
        'batch_size': cfg.training.batch_size,
        'num_workers': cfg.training.num_workers,
        'pin_memory': cfg.training.pin_memory,
        'flip_channel_prob': cfg.augmentation.flip_channel_prob,
        'mask_vad_prob': cfg.augmentation.mask_vad_prob,
        'mask_vad_scale': cfg.augmentation.mask_vad_scale,
        'mode': cfg.training.mode,
    })
    
    # Dynamically instantiate DataModule based on config
    datamodule_class_name = cfg.data['datamodule_class']
    DataModuleClass = getattr(mmvap_data, datamodule_class_name)
    print(f"Using {datamodule_class_name} for dataset: {cfg.data['name']}")
    datamodule = DataModuleClass(data_cfg)
    
    save_dir = cfg.paths.get('save_dir')
    prefix_save = cfg.paths.get('prefix_save')
    prefix_suffix = cfg.paths.get('prefix_suffix', '')
    if prefix_suffix:
        prefix_save = f"{prefix_save}{prefix_suffix}"
    logtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(save_dir, prefix_save, logtime)

    # merge model_cfg and data_cfg so that the saved hparams match the old format
    # merged_cfg = {**model_cfg, **data_cfg, **cfg}
    # print(f'merged_cfg: {merged_cfg}')
    model = MMVAPLightningModule(cfg) # pass in cfg rather than model_cfg to match the original codebase's format.
    ## check grads
    print(f'non-frozen ones:')
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    ## end check grads
    # Load model pretrained weights if specified
    pretrained_checkpoint = cfg.paths.get('pretrained_checkpoint')
    os.makedirs(os.path.join(cfg.paths.checkpoint_dir, prefix_save), exist_ok=True)
    if pretrained_checkpoint is not None:
        print(f"\nLoading pretrained checkpoint: {pretrained_checkpoint}")
        model = load_pretrained_weights(model, pretrained_checkpoint)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, prefix_save, logtime, cfg.paths.checkpoint_dir),
        filename='{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    logger = TensorBoardLogger(
        save_dir=os.path.join(save_dir, prefix_save, logtime, cfg.paths.log_dir),
        name=f"{cfg.model.model_name}_{cfg.model.feature_set}",
        version=None,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, lr_monitor, ModelSummary(max_depth=3)],
        logger=logger,
        deterministic="warn",
    )
    
    # Train
    print("Starting training...")
    print(f'Logging to: {os.path.join(save_dir, prefix_save, logtime, cfg.paths.log_dir)}')
    print(f"Checkpoints: {os.path.join(save_dir, prefix_save, logtime, cfg.paths.checkpoint_dir)}")
    print("="*80 + "\n")
    if pretrained_checkpoint:
        print(f'attempting to resume from pretrained checkpoint: {pretrained_checkpoint}')
        trainer.fit(model, datamodule=datamodule, ckpt_path=pretrained_checkpoint if pretrained_checkpoint and pretrained_checkpoint.endswith('.ckpt') else None if pretrained_checkpoint and pretrained_checkpoint.endswith('.ckpt') else None) # when ckpt crashes
    else:
        trainer.fit(model, datamodule=datamodule)
    
    # Save hparams to checkpoint directory as well
    import yaml
    import json
    checkpoint_dir = os.path.join(save_dir, prefix_save, logtime, cfg.paths.checkpoint_dir)

    # Save original Lightning config as well (in order to reproduce stuffs)
    hparams_lightning_path = os.path.join(checkpoint_dir, "hparams_lightning.yaml")
    with open(hparams_lightning_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False)
    print(f"Saved original Lightning config to: {hparams_lightning_path}")
    
    # Print best checkpoint
    print("\n" + "="*80)
    print("Training completed!")
    print(f'Hparams logged to: {os.path.join(save_dir, prefix_save, logtime, cfg.paths.log_dir)}/{cfg.model.model_name}_{cfg.model.feature_set}/version_0/hparams.yaml')

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/loss: {checkpoint_callback.best_model_score:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()

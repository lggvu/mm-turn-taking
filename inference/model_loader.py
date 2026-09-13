"""
Load a VAP / multimodal-VAP checkpoint and instantiate the matching model
class automatically -- no separate config file needed.

Checkpoints in this project's history store their build config under
`hyper_parameters` in a few different shapes:

  1. "new":  {model_name, device, encoder, base, multimodal, feature_set, ...}
  2. "old":  {model_cfg, multimodal_model_cfg, encoder_cfg, device,
              feature_set, training_cfg}  (already the shape the raw model
              classes' __init__ expects)
  3. {"cfg": <old shape>, "model_name": ..., "feature_set": ...}

If hyper_parameters don't say which class was used (shape 2 never does),
we fall back to (a) matching a known class name inside the checkpoint path,
then (b) trying every candidate class and keeping whichever one loads the
state dict with zero missing/unexpected keys.
"""
import re
from pathlib import Path

import torch
import yaml

from mmvap.models.model import StereoTransformerModel, StereoTransformerModelVideoOnly
from mmvap.models.multimodal_model import EarlyVAFusion, LateVAFusion

MODEL_CLASSES = {
    "EarlyVAFusion": EarlyVAFusion,
    "LateVAFusion": LateVAFusion,
    "StereoTransformerModelVideoOnly": StereoTransformerModelVideoOnly,
    "StereoTransformerModel": StereoTransformerModel,
}

# Longest names first so "StereoTransformerModelVideoOnly" isn't missed by
# an early substring match against "StereoTransformerModel".
_NAME_SEARCH_ORDER = sorted(MODEL_CLASSES, key=len, reverse=True)

# Which inputs each model class's forward() actually consumes.
MODALITY = {
    "StereoTransformerModel": {"audio": True, "video": False},
    "StereoTransformerModelVideoOnly": {"audio": False, "video": True},
    "EarlyVAFusion": {"audio": True, "video": True},
    "LateVAFusion": {"audio": True, "video": True},
}

_CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "model"
_DEFAULT_CONFIG_FILES = {
    "EarlyVAFusion": "early_va_fusion.yaml",
    "LateVAFusion": "early_va_fusion.yaml",
    "StereoTransformerModel": "vap.yaml",
    "StereoTransformerModelVideoOnly": "video.yaml",
}


def _deep_unwrap_omegaconf(obj):
    """Recursively convert OmegaConf nodes to plain Python values.

    Some older checkpoints saved their hyperparameters as (partially)
    OmegaConf DictConfig/AnyNode objects rather than plain dicts. Depending
    on which OmegaConf version wrote them, `OmegaConf.to_container`/`dict()`
    on the *currently installed* OmegaConf can itself blow up on corrupted
    key_type metadata (`issubclass() arg 1 must be a class`) -- so instead
    of relying on OmegaConf's own accessors, walk each node's private
    `_content`/`_val` directly, which is version-stable.
    """
    content = getattr(obj, "_content", None)
    if content is not None:
        return _deep_unwrap_omegaconf(content)
    if hasattr(obj, "_val"):
        return _deep_unwrap_omegaconf(obj._val)
    if isinstance(obj, dict):
        return {k: _deep_unwrap_omegaconf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_unwrap_omegaconf(v) for v in obj]
    return obj


def _default_hp_for(model_name: str) -> dict:
    """This project's shipped default hyperparameters for a model class,
    used only when a checkpoint carries no hyper_parameters of its own."""
    fname = _DEFAULT_CONFIG_FILES.get(model_name, "early_va_fusion.yaml")
    with open(_CONFIG_ROOT / fname) as f:
        return yaml.safe_load(f)


def _new_hp_to_old_cfg(hp: dict, device: str) -> dict:
    old_cfg = {}
    if "multimodal" in hp:
        old_cfg["multimodal_model_cfg"] = hp["multimodal"]
    if "base" in hp:
        old_cfg["model_cfg"] = dict(hp["base"])
    encoder = hp.get("encoder", {}) or {}
    old_cfg["encoder_cfg"] = {"class": encoder.get("class", "StereoEncoder"), "params": {}}
    old_cfg["encoder_info"] = {
        "audio_sample_rate": encoder.get("audio_sample_rate", 16000),
        "sample_rate": encoder.get("feature_sample_rate", 50),
    }
    old_cfg["device"] = device
    old_cfg["feature_set"] = hp.get("feature_set", "all")
    return old_cfg


def _extract_old_cfg_and_name(hp: dict, device: str):
    """Best-effort (old_cfg, model_name) from a checkpoint's hyper_parameters.
    Either element may come back None."""
    if not hp:
        return None, None

    if isinstance(hp.get("cfg"), dict) and "model_cfg" in hp["cfg"]:
        return hp["cfg"], hp.get("model_name")

    if "model_cfg" in hp:
        # Already old-format; encoder_cfg here may be missing class/params.
        old_cfg = dict(hp)
        encoder_cfg = dict(old_cfg.get("encoder_cfg") or {})
        encoder_cfg.setdefault("class", "StereoEncoder")
        encoder_cfg.setdefault("params", {})
        old_cfg["encoder_cfg"] = encoder_cfg
        return old_cfg, hp.get("model_name")

    if "base" in hp or "multimodal" in hp:
        return _new_hp_to_old_cfg(hp, device), hp.get("model_name")

    return None, None


def _guess_name_from_path(checkpoint_path: str):
    text = str(checkpoint_path)
    for name in _NAME_SEARCH_ORDER:
        if re.search(name, text):
            return name
    return None


def _guess_name_by_trial_load(old_cfg: dict, state_dict: dict):
    """Instantiate every candidate class with old_cfg and try loading
    state_dict into it; return whichever name matches with the fewest
    missing/unexpected keys (None if every candidate errors out)."""
    best_name, best_score = None, None
    for name, cls in MODEL_CLASSES.items():
        try:
            model = cls(old_cfg)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except Exception:
            continue
        score = len(missing) + len(unexpected)
        if score == 0:
            return name
        if best_score is None or score < best_score:
            best_name, best_score = name, score
    return best_name


def _strip_prefix(state_dict: dict) -> dict:
    prefix = "model."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def load_model(checkpoint_path: str, device: str = "cuda:0", model_class: str = None):
    """Load a VAP / multimodal-VAP model straight from a training checkpoint.

    Args:
        checkpoint_path: path to a PyTorch Lightning .ckpt (or a raw
            state-dict .pt) saved by scripts/train.py.
        device: torch device string.
        model_class: optional override, one of MODEL_CLASSES, in case
            auto-detection guesses wrong or the checkpoint has no metadata.

    Returns:
        (model, model_name, mode, modality) -- `mode` is 'VAP'/'ind-4'/'ind-40',
        `modality` is {'audio': bool, 'video': bool} for which inputs the
        model needs.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        hp = ckpt.get("hyper_parameters", {}) or {}
    else:
        state_dict = ckpt
        hp = {}
    hp = _deep_unwrap_omegaconf(hp)
    state_dict = _strip_prefix(state_dict)

    old_cfg, model_name = _extract_old_cfg_and_name(hp, device)

    if model_class:
        model_name = model_class
    elif model_name is None:
        model_name = _guess_name_from_path(checkpoint_path)

    if old_cfg is None:
        # No usable hyper_parameters at all -- fall back to this project's
        # shipped default config for whatever model class we can identify.
        guessed = model_name or "EarlyVAFusion"
        old_cfg = _new_hp_to_old_cfg(_default_hp_for(guessed), device)
        print(f"[infer] checkpoint has no build config; using this repo's default "
              f"'{guessed}' hyperparameters. Pass model_class=... if that's wrong.")

    if model_name is None:
        model_name = _guess_name_by_trial_load(old_cfg, state_dict)
        if model_name is None:
            raise ValueError(
                f"Could not determine the model class for checkpoint '{checkpoint_path}'. "
                f"Pass model_class=... explicitly (one of {list(MODEL_CLASSES)})."
            )
        print(f"[infer] model class not recorded in checkpoint; inferred '{model_name}' "
              f"by trial-loading its weights.")

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model class '{model_name}'. Expected one of {list(MODEL_CLASSES)}.")

    old_cfg.setdefault("device", device)
    model = MODEL_CLASSES[model_name](old_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[infer] warning: loaded '{model_name}' with {len(missing)} missing and "
              f"{len(unexpected)} unexpected keys "
              f"(missing[:5]={missing[:5]}, unexpected[:5]={unexpected[:5]})")

    model = model.to(device)
    model.eval()

    mode = old_cfg.get("model_cfg", {}).get("mode", "VAP")
    modality = MODALITY[model_name]
    print(f"[infer] loaded {model_name} (mode={mode}, needs_audio={modality['audio']}, "
          f"needs_video={modality['video']}) on {device}")
    return model, model_name, mode, modality

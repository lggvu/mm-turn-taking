# MM-Turn-Taking

Audio-VAP, Video-VAP and MM-VAP: multimodal Voice Activity Projection (VAP)
models for turn-taking prediction, trained and evaluated on the **CANDOR**
and **AVCocktail** conversational datasets.

- **Audio-VAP** (`StereoTransformerModel`) -- audio-only.
- **Video-VAP** (`StereoTransformerModelVideoOnly`) -- video-only.
- **MM-VAP** (`EarlyVAFusion`) -- early audio-visual fusion.

This is a public release of the training/eval code used for the
paper: PyTorch Lightning + Hydra training, and standalone evaluation scripts
for shift/hold prediction. All released recipes use the CPC audio encoder
(weights downloaded automatically on first use, see
`mmvap/audio_encoders/encoder_cpc_components.py`); no other encoder is
supported in this release.

## Structure

```
mm-turn-taking/
├── configs/                # Hydra configs (model / data / encoder / trainer)
├── mmvap/                  # Package: models, data modules, audio encoder
├── scripts/train.py        # PyTorch Lightning + Hydra training entrypoint
├── eval/
│   ├── evaluate_dataset.py       # AVCocktail shift/hold evaluation
│   └── evaluate_candor_folds.py  # CANDOR (k-fold) shift/hold evaluation
└── recipes/                # One reproducible recipe per released checkpoint
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python >= 3.8 and a CUDA-capable GPU for training/inference at
realistic speed (CPU works but is slow).

The codebase was built initially in Python 3.8, if you use a newer Python and hydra-core (Python >3.8 and hydracore > 1.0.7), you may need to add `version_base=None` to this line in `scripts/train.py`:
```
# Old: @hydra.main(config_path="../configs", config_name="train_config")
@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
```


## Dataset setup

`configs/data/avcocktail.yaml` and `configs/data/candor.yaml` ship with
`/path/to/...` placeholder paths -- point them at your local copies of the
datasets (or override on the command line, e.g. `data.train.wavdir=...`).
Neither dataset is redistributed here.

## Training

```bash
# From scratch / with your own hyperparameters:
python scripts/train.py model=vap data=avcocktail encoder=cpc

# To exactly reproduce a released checkpoint, pass its recipe config instead:
python scripts/train.py --config-path=../recipes --config-name=audio-vap-avcocktail
```

`model` selects the architecture (`vap` = Audio-VAP, `video` = Video-VAP,
`early_va_fusion` = MM-VAP), `data` selects the dataset (`avcocktail` or
`candor`). See `recipes/` for a complete, standalone config -- model +
dataset + encoder + hyperparameters -- per released checkpoint.

## Evaluation

```bash
cd eval
python evaluate_dataset.py model.config_path=... model.weights_path=...        # AVCocktail
python evaluate_candor_folds.py model.config_path=... model.weights_path=...   # CANDOR
```

Both scripts are Hydra-based (`eval/conf/config_avcocktail.yaml` /
`config_candor.yaml`); override dataset paths, model paths, and evaluation
window/threshold on the command line as shown in `recipes/`.

## Recipes

See [recipes/README.md](recipes/README.md) for the full list of reproducible
training + evaluation recipes, one per released checkpoint.

## License

MIT, see [LICENSE](LICENSE). The CPC encoder weights are downloaded from the
public [CPC_audio](https://github.com/facebookresearch/CPC_audio) checkpoint
release; the CANDOR and AVCocktail datasets are governed by their own
respective licenses/access terms and are not redistributed here.

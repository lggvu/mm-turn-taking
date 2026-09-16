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
├── avcocktail_prep/        # AVCocktail prep codes for turn-taking with VAP
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

## Turn-taking Labels
Turn-taking labels are available at: https://huggingface.co/datasets/lggvu/avcocktail-turn-taking.

## Setup AVCocktail
Refer to `avcocktail_prep` to download and process the data for training/evaluation.
The turn labels are available at: https://huggingface.co/datasets/lggvu/avcocktail-turn-taking  
The pre-extracted OpenFace features are available at: https://huggingface.co/datasets/lggvu/avcocktail-openface-features  
You can also extract the features yourself with an OpenFace docker image as in `avcocktail-prep/openface_connector.py`.  
At the end you should have the following components for training/evaluation:  

- Stereo audios of two speakers
- OpenFace features of speakers (for video-based models)  
- Chunks generated in .pt and .npy files, which are defined by window size and hop size constants in the bash scripts. These are used to speed up the training.  
- Event files (`.json`) as labels.

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
Model make predictions at 50Hz. A shift is predicted if the VAP probability of the *other* speaker is higher than a threshold. Since the probability is cumulated over a 2-second window (thus 2*50=100 predictions), a baseline threshold (0.5) will be 50.0.

```bash
cd eval
python evaluate_dataset.py model.config_path=... model.weights_path=...        # AVCocktail
python evaluate_candor_folds.py model.config_path=... model.weights_path=...   # CANDOR
```

Both scripts are Hydra-based (`eval/conf/config_avcocktail.yaml` /
`config_candor.yaml`); override dataset paths, model paths, and evaluation
window/threshold on the command line as shown in `recipes/`. 
Note that the config_path should be the `hparams.yaml` file containing only the model's parameters, not the `hparams_lightning.yaml` files which contain more than you need. I've never tried running evaluation with those.

## Recipes

See [recipes/README.md](recipes/README.md) for the full list of reproducible
training + evaluation recipes, one per released checkpoint.

## Pre-trained checkpoints
| Model | Trained on | Hugging Face |
|---|---|---|
| MMVAP | Candor | [lggvu/mmvap-candor](https://huggingface.co/lggvu/mmvap-candor) |
| MMVAP | Candor → AVCocktail | [lggvu/mmvap-candor-avcocktail-ft](https://huggingface.co/lggvu/mmvap-candor-avcocktail-ft) |
| MMVAP | AVCocktail | [lggvu/mmvap-avcocktail](https://huggingface.co/lggvu/mmvap-avcocktail) |
| AudioVAP | Candor | [lggvu/audiovap-candor](https://huggingface.co/lggvu/audiovap-candor) |
| AudioVAP | Candor → AVCocktail | [lggvu/audiovap-candor-avcocktail-ft](https://huggingface.co/lggvu/audiovap-candor-avcocktail-ft) |
| AudioVAP | AVCocktail | [lggvu/audiovap-avcocktail](https://huggingface.co/lggvu/audiovap-avcocktail) |
| VideoVAP | Candor | [lggvu/videovap-candor](https://huggingface.co/lggvu/videovap-candor) |
| VideoVAP | Candor → AVCocktail | [lggvu/videovap-candor-avcocktail-ft](https://huggingface.co/lggvu/videovap-candor-avcocktail-ft) |
| VideoVAP | AVCocktail | [lggvu/videovap-avcocktail](https://huggingface.co/lggvu/videovap-avcocktail) |


"A->B" means pre-trained on A, then fine-tuned on B.

## License

MIT, see [LICENSE](LICENSE). The CPC encoder weights are downloaded from the
public [CPC_audio](https://github.com/facebookresearch/CPC_audio) checkpoint
release; the CANDOR and AVCocktail datasets are governed by their own
respective licenses/access terms and are not redistributed here.

## Citation
Accepted to IEEE SLT 2026. Citations TBU.

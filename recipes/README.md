# Recipes

Each `.yaml` file here is a complete, standalone, reproducible Hydra config
for one released checkpoint -- a fully composed config (base training config
+ model + dataset + encoder + any per-run overrides), not a fragment. Pass it
straight to the training script:

```bash
python scripts/train.py --config-path=../recipes --config-name=audio-vap-avcocktail
```

(`--config-path` is relative to `scripts/`, matching how the script resolves
its own default `../configs`; an absolute path also works.) Each file's
header comment documents what it is and gives the matching
`eval/evaluate_dataset.py` command. All recipes use the CPC audio encoder --
no other encoder is supported by this release.

| Recipe | Model | Trained on |
|---|---|---|
| [audio-vap-avcocktail.yaml](audio-vap-avcocktail.yaml) | Audio-VAP (`StereoTransformerModel`) | AVCocktail only |
| [audio-vap-candor-ft-avcocktail.yaml](audio-vap-candor-ft-avcocktail.yaml) | Audio-VAP | CANDOR &rarr; fine-tuned on AVCocktail |
| [video-vap-avcocktail.yaml](video-vap-avcocktail.yaml) | Video-VAP (`StereoTransformerModelVideoOnly`) | AVCocktail only |
| [video-vap-candor-ft-avcocktail.yaml](video-vap-candor-ft-avcocktail.yaml) | Video-VAP | CANDOR &rarr; fine-tuned on AVCocktail |
| [mm-vap-avcocktail.yaml](mm-vap-avcocktail.yaml) | MM-VAP (`EarlyVAFusion`) | AVCocktail only |
| [mm-vap-candor-ft-avcocktail.yaml](mm-vap-candor-ft-avcocktail.yaml) | MM-VAP | CANDOR &rarr; fine-tuned on AVCocktail |

## Dataset setup

Every recipe ships with `/path/to/...` placeholder paths under `data.train` /
`data.val` -- edit them in the file, or override on the command line as shown
above, before running.

## CANDOR pretraining stage (prerequisite for the `*-candor-ft-avcocktail` recipes)

The three `*-candor-ft-avcocktail` recipes fine-tune from a checkpoint
trained on CANDOR first. This release does not include a dedicated recipe
file for that first stage; use the corresponding `*-avcocktail` recipe with
`data=candor` substituted in, e.g. for Audio-VAP:

```bash
python scripts/train.py --config-path=../recipes --config-name=audio-vap-avcocktail \
    data=candor paths.prefix_save=Audio-VAP-candor
```

Point the fine-tuning recipe's `paths.pretrained_checkpoint` (already present
as a `/path/to/...` placeholder in each `*-candor-ft-avcocktail.yaml`) at the
resulting `save/Audio-VAP-candor/<timestamp>/checkpoints/last.ckpt` (or a
specific epoch checkpoint).

# Matched 3D PlaquePilot experiments

Run in the existing PlaquePilot branch at commit `72091c16369becd6b96e1756b6dd3ab353270aac`. Extract the ZIP directly into `D:\Code\ImageCAS-X`. The archive overlays the native trainer and CLI and adds the launcher, protocol, model implementations, and historical fusion evaluation. It includes the upstream MedNeXt-S architecture and license. Swin UNETR v2 uses MONAI with `use_v2=True`, 24 initial features, and activation checkpointing.

## From Anaconda Prompt

Verify the branch, look for existing local edits, then extract once. The default download path below assumes the usual Windows Downloads directory.

```bat
conda activate PlaquePilot
cd /d D:\Code\ImageCAS-X
git branch --show-current
git status --short
python -m zipfile -e "%USERPROFILE%\Downloads\PlaquePilot_3D_matched_long.zip" .
git status --short
```

Preserve any local edits to `plaque_benchmark/native.py` or `plaque_benchmark/__main__.py` before overlaying those two tracked files. The prior ZIPs, a branch switch, and a separate MedNeXt checkout are unnecessary. Run from the repository root using the same CUDA environment and CTTA installation as the previous experiments.

```bat
python run_PlaquePilot_3D.py check
python run_PlaquePilot_3D.py init
```

If imports are missing, install the repo’s existing plaque dependencies with `python -m pip install -e ".[plaque]"` and its pinned CTTA/search requirements with `python -m pip install -r requirements_workstation.txt`. For a missing or incompatible MONAI, use `python -m pip install --no-deps monai==1.5.2` and `python -m pip install einops`. Keep the working CUDA PyTorch build. The check verifies the `use_v2` constructor, CUDA, cohort identities, prior approval receipt, input files, software and shared protocol. The fresh initialization trains each model, validates it, verifies finite losses and plaque patch Dice, and reopens the checkpoints.

After all three initialization checks pass, run:

```bat
python run_PlaquePilot_3D.py run
```

This runs the search for each architecture, trains its selected recipe, scores the fixed-budget comparison on validation, continues the validation-winning architecture, and scores its extended checkpoint. The single command from a new extraction is `python run_PlaquePilot_3D.py all`, which includes initialization. For a second matched comparison of all three extended networks, use `python run_PlaquePilot_3D.py run --extend-all` after initialization. Each extended arm then has the same cap and patience but can stop at a different epoch. Choose the extension policy before starting that stage. The default `all` and `run` finish at validation.

Run individual phases when desired:

```bat
python run_PlaquePilot_3D.py tune
python run_PlaquePilot_3D.py train
python run_PlaquePilot_3D.py score
python run_PlaquePilot_3D.py extend
```

Rerun the interrupted command with the same options. Completed trials have individual receipts. The Optuna grid is reconstructed from those scores on restart. Interrupted training resumes with the stored optimizer, scheduler and epoch. The launcher prevents a second process from writing to the same experiment root. A changed protocol or software/source identity requires a new output folder and fresh initialization. If the common microbatch of four exceeds the A5000 budget, choose batch 2 and accumulation 16 **for every arm**, then use a new `--out` consistently through all stages. This keeps 32 patches per update but changes the microbatch Dice calculation.

## What is matched

| Setting | Common design |
|---|---|
| Cohort | Original accepted physical-space manifest, same file and split hashes. 75 train, 16 validation, 16 held out. |
| Task and inputs | Native 3D CCTA. One CT channel and three segmentation classes. Same HU processing, original physical-space masks and class mapping. |
| Sampling | 64³ voxels, 32 patches per update (4 × 8 accumulation), 100 updates, 3,200 training draws per epoch from all 75 patients. 96 fixed validation patches from all 16 validation patients. |
| Patch targets and CTTA | Historical tuned FFR/Swin v1 mixture. Plaque 37.94%, healthy branch 15%, foreground 25%, remainder random. CTTA step 16.65% and step-motion 10.16%. Training-only augmentation. |
| Optimization controls | Same seed 2028, Dice + cross-entropy loss, weight decay 2.2834e-5, clipping, mixed-precision policy, cosine schedule, validation frequency and inference overlap 0.5. |
| Initialization | One full epoch per architecture with common historical AdamW/LR. Full training starts afresh after tuning. |
| Optuna | GridSampler exhausts the same four optimizer/LR combinations per architecture. Each trial runs four complete epochs (12,800 draws), with pruning and early stopping disabled. AdamW LRs 4.6712338138743875e-5 and 3e-4. SGD with Nesterov momentum 0.9, LRs 0.003 and 0.015. The objective is peak validation plaque patch Dice over the four epochs. Ties resolve by optimizer name then smaller LR. |
| Fixed comparison | Each selected configuration trains exactly 120 epochs (12,000 updates, 384,000 draws) from fresh weights. Each network gets the same 240-epoch cosine horizon to permit later continuation. Save the best plaque patch Dice checkpoint within the first 120 epochs. |
| Promotion | Rank all three checkpoints by full-volume mean plaque Dice on plaque-positive validation patients. A tie resolves by model ID. Each architecture’s search winner is distinct from the architecture chosen for extension. |
| Extension | Continue from epoch 120’s last weights, optimizer and scheduler. Up to 240 total (120 additional). Patience 35 begins after the common 120-epoch boundary. Improvements are measured against the best validation plaque patch Dice seen so far. Earlier best weights remain eligible. |
| Full-volume scoring | Same argmax segmentation and evaluator, 1 mm surface tolerance, one-to-one lesion matching requiring positive overlap. Report plaque/lumen Dice, precision/recall, healthy-branch false positives, lesion and surface metrics. Paired Dice differences vs fresh FFR control use patient bootstrap intervals; these do not estimate variability across training seeds. |

The historical native runs used an 80-epoch cap and patience 7. The new cap, stopping and search rules apply prospectively to all three arms. The FFR control measures the effect of running that architecture under the new protocol. Historical external nnU-Net and ResEnc used their own 3D plans and sampling rules. They are contextual baselines. Search selection is best among four prespecified candidates on validation. All new primary comparisons use the same data and patch size, including the same case ordering and crop seed per epoch.

## Outputs and scoring

Default root: `experiments\PlaquePilot_confirmed\runner\PlaquePilot_3D_matched120_long240`.

- `protocol_lock.json` stores exact patient lists, source and dataset hashes, package versions, GPU, architecture settings, and common protocol.
- `init\<model>\training.log` and `PASS.json` record initialization. Each stage has training logs, checkpoint files and saved history.
- `search\<model>\trials.csv`, `winner.json` and `best_config.json` provide search outcomes and exported settings.
- `matched\<model>\stage.json` and `history.csv` retain the frozen matched-budget snapshot. `long\<model>` records the continuation separately. Live resume checkpoints are in `training\<model>\model`.
- `scoring\val\matched\comparison\endpoints_and_training.csv` contains measured performance, checkpoint identities, selected epochs, actual updates, draws, and summed epoch time. `paired_dice_vs_ffr.csv` and `comparison.png` give paired and visual comparisons. `historical_validation_context.csv` includes compatible historical reports present on disk. `matched_selection.json` identifies the model promoted.
- `scoring\val\long\comparison` contains extended results. Cases, lesions and surface metrics are under the corresponding `reports\<model>` folders.

After the validation decisions are final, freeze the candidate checkpoints, then run held-out scoring explicitly:

```bat
python run_PlaquePilot_3D.py freeze
python run_PlaquePilot_3D.py test
```

The held-out comparisons are in `scoring\test\matched\comparison` and `scoring\test\long\comparison`. Report the fixed-budget table for architecture comparisons and the extension table for the longer-training result. Reusing a held-out result to change the recipe turns that split into development data.

## Historical specialist and ensemble check

To replay archived CTTA-on standard nnU-Net, ResEnc and FFR checkpoints, including ResEnc lumen + FFR plaque, run:

```bat
python run_PlaquePilot_3D.py legacy-val
```

This is a validation-only replay of the previous specialist, ensemble and filtering rules. It requires the original checkpoints or archived predictions plus the checkpoint identities, the recorded nnU-Net plans, and the previous nnU-Net installation. When necessary, use `--nnunet-results D:\Code\nnUNet\results` or pass explicit checkpoint paths with `--nnunet`, `--nnunet-resenc` and `--ffr-unet`. View `historical_confirmation\val\comparison\confirmation_metrics.csv` and the paired detection/dice tables.

## Record the package on PlaquePilot

After verifying the overlay, these commands record just the supplied code on the existing branch. This repository ignores new JSON files, so explicitly stage the experiment protocol.

```bat
git add PLAQUEPILOT_3D_RUN.md run_PlaquePilot_3D.py plaque_benchmark/native.py plaque_benchmark/__main__.py plaque_benchmark/round2_models.py plaque_benchmark/round2_confirm.py plaque_benchmark/mednext_v1
git add -f plaque_3d_protocol.json
git diff --cached --stat
git commit -m "Add matched 3D architecture tuning and extended plaque training"
```

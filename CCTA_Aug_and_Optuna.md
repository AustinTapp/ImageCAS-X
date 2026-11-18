# ImageCAS-X workstation, CTTA_Aug and Optuna update

This is an overlay for the extracted ImageCAS-X plaque benchmark plus the previous
training-speed patch. It includes the speed patch's native trainer and launcher.
Stop active training before replacing code files. Keep the existing baseline
experiment and checkpoints. Augmentation/hyperparameter changes are NEW runs;
do not resume the old baseline under a new augmentation policy.

## Results from your latest re-init

All five networks passed one epoch, 100 updates, finite losses and checkpoint checks.

| Network | Epoch seconds | Peak reserved GiB | Native patches/second, including init validation |
|---|---:|---:|---:|
| CAS-Net | 60.35 | 9.63 | 53.02 |
| FFR-UNet | 221.06 | 9.50 | 14.48 |
| Swin UNETR | 196.42 | 9.57 | 16.29 |
| nnU-Net | 82.82 | 21.22 | Different planned patches; not directly comparable |
| Residual-encoder nnU-Net | 112.35 | 21.56 | Different planned patches; not directly comparable |

The first 91-case cache build took about 10m45s, outside the epoch timing. Later
models reused the cache. Native loading wait was ~6.5-7.3s per initialization
including worker startup; it barely increased after startup. The earlier two-hour
CAS-Net epochs were input-bound. The new run is primarily doing model work.
Full training validates 96 patches instead of the smaller init validation budget,
so the 60-second initialization epoch is not an exact prediction of full epoch time.

Keep nnU-Net batch=2. Its actual peak is already near the 22 GiB cap. Do not set
--nnunet-batch-size 4 on the basis of the older desktop memory observation.
Both nnU-Nets had zero foreground pseudo Dice after the single init epoch. This
is not an execution failure or an accuracy validation. Follow the learning curves
and inspect full-volume predictions if it persists. The torch.onnx architecture
plot warning is cosmetic: training/checkpoint verification completed afterward.

## Install

Extract the ZIP into D:\Code\ImageCAS-X, replacing matching code files. Existing
experiment data, patient splits and checkpoints are not included or overwritten.
Keep the previous source snapshot if continuing the old baseline is needed.

In Anaconda Prompt:

```bat
conda activate omni
cd /d D:\Code\ImageCAS-X
python -m pip install -r requirements_workstation.txt
set "PLAQUE_PYTHON=C:\Users\austin\miniconda3\envs\omni\python.exe"
set "GIT_BASH=C:\Program Files\Git\bin\bash.exe"
```

CTTA_Aug is pinned to AustinTapp/CTTA_Aug commit
7fbebab4d391f45432f92e8374adaa2770d986eb (ctaug 0.1.2). This overlay does not vendor
or replace that project's source. The requirements do not request a PyTorch
upgrade; retain your working CUDA installation.
If NNUNET_PYTHON points to another environment, install the same requirements in
that environment too using its Python executable. CTAug code/version hashes are
recorded in the launcher environment probe and initialization fingerprint.

## Hardware configuration

Use --a5000 --workstation for new runs. Native physical batches remain CAS 32,
FFR 8 and Swin 4, with accumulation 1/4/8, respectively: effective batch 32.
Patches remain 64 cubed. The 22 GiB allocator cap remains in place.

--workstation uses four persistent native loading workers, two prefetched batches
per worker, four parallel case-cache builders, and a single PyTorch/ITK compute
thread per worker. Native per-worker caches contain read-only mappings, not dense
copies of every CT. Windows can share cached file pages across workers and models,
using your available RAM without making four separate full-RAM caches. Runtime
logical CPU count and available RAM are recorded in environment.json; no CPU core
count is guessed from the processor model number.

The external nnU-Net augmentation process default is six with --workstation.
An existing nnUNet_n_proc_DA environment setting takes precedence. The default
nnU-Net plans, spacing, patch sizes and batch=2 are preserved.

Four cache builders can transiently hold several decoded cases. Leave headroom
for other applications; avoid concurrent GPU trials or full-dataset RAM copies.
Disk-cache builds and reuse keep the same content-hash keys as the speed patch.
The cache remains reusable across different random augmentation settings.
Use an SSD for PLAQUE_CACHE_DIR if the default prepared/native_patch_cache folder
is on slow storage. The OS releases cached pages under memory pressure.

More CPU workers are not guaranteed to help. For a separate benchmark run,
--workers 2 or --workers 8 overrides the native profile. Compare post-startup
train_patches_per_second and data_wait_seconds with identical sample/update
budgets. Do not alter workers/configuration in the middle of a resumed run.

## Training with CTTA_Aug

The mild profile applies StepTransform at p=0.10 and StepMotionAugmentation at
p=0.10. The plain MotionAugmentation path is also implemented and configurable.
All execute on raw-HU crops in loading workers, after random patch selection and
before native HU normalization. NIfTI voxel XYZ arrays and spacing are transposed
together into the ZYX convention expected by CTAug and transposed back afterward.
This assumes the native CT's third voxel axis is its acquisition slice axis, as
in the supplied dataset. Review previews for differently oriented acquisitions.

Motion is paired for image, target and validity. This adapter deliberately uses
the upstream functional implementation with zero fill and without the wrapper's
random cut-and-resize mode. This preserves small vessel/plaque label IDs. Newly
exposed native target voxels use ignore=255. There is no interpolation of class IDs.

The nnU-Net adapter is prepended to its standard training augmentation pipeline,
before spatial augmentation/deep supervision. It reconstructs HU from the plan's
CTNormalization mean/std, applies the same CTTA policy in preprocessed ZYX spacing,
and normalizes again. The input has already undergone nnU-Net intensity clipping;
this is not restoration of intensities removed by preprocessing. The adapter
requires one CT channel and the standard scalar multiclass target. It uses -1
for padding; nnU-Net's existing transforms subsequently map padding to background
as in the unmodified trainer. Validation transforms are not wrapped. Standard
nnU-Net augmentations remain enabled, so this is a CTTA addition to each method's
baseline, not proof of an identical complete augmentation distribution.

MetalTransform, WireTransform and CalcificationTransform are integrated but
DISABLED in the mild profile. They forward-project/reconstruct affected slices
and can be expensive. They insert synthetic objects without constructing new
disease/device annotations. The experimental profile explicitly accepts that
these are image-corruption robustness transforms with original underlying labels:
- calcification anchors on existing plaque label 2;
- wire/metal anchor on valid background, not on the lumen;
- per-transform probability is 0.05 and at most one implant specification is used;
- severity is 0.10-0.30.
This is not synthetic plaque ground-truth generation. Separate vessel annotations
cannot provide chamber-specific device-placement guidance. Visually review this
experimental ablation before using it for final comparison. No robustness gain on
coronary plaque segmentation is claimed from the upstream cardiac-anatomy study.

### Optional preview

Choose a TRAINING case ID from prepared/dataset.json:

```bat
python -m plaque_benchmark.ctta --dataset experiments/plaque_all_accepted_physical/prepared/dataset.json --config configs/ctta_mild.json --case SHK0001 --out experiments/ctta_preview --seed 2028
```

Replace SHK0001 if it is not in your training partition. Inspect the original and
augmented CT/label NIfTIs with the recorded JSON transform parameters. A seed can
sample no transforms with a probabilistic policy; change the seed or use a preview
configuration with a chosen transform probability=1. Do not tune on test cases.

### New augmented five-network comparison

No dataset re-preparation is required:

```bat
"%GIT_BASH%" run_plaque_benchmark.sh init --bench experiments/plaque_all_accepted_physical --run-name workstation_ctta25 --a5000 --workstation --ctta-config configs/ctta_mild.json
"%GIT_BASH%" run_plaque_benchmark.sh train --bench experiments/plaque_all_accepted_physical --run-name workstation_ctta25 --a5000 --workstation --ctta-config configs/ctta_mild.json
```

Run train only after all init checks pass. Reuse every option on resume and add
--resume. Use a separate run name if choosing the experimental object profile.
The existing a5000_screen25 results are the non-CTTA baseline. This overlay adds
new modules/source fingerprints and new training settings, so the old
--adopt-performance-update exception is intentionally not sufficient to migrate
that run to the augmented experiment.

## Optional Optuna search

This first integration tunes NATIVE models (CAS-Net, FFR-UNet, Swin UNETR and optional
DynUNet). External nnU-Net augmentation is integrated; nnU-Net hyperparameter search
is NOT implemented. Do not describe the complete five-model benchmark as tuned
by Optuna. The current external nnU-Net configurations remain planned baselines.

Run this separately from any active training. One child training process/GPU trial
runs at a time; Optuna itself uses n_jobs=1. A local file lock prevents overlapping
search coordinators sharing the output parent. No Ray, Dask, or parallel GPU trials
are started. CPU loading and augmentation remain parallel inside a trial.

Example (CAS-Net first, then expand --models if the budget is acceptable):

```bat
python -m plaque_benchmark.optuna_search --dataset experiments/plaque_all_accepted_physical/prepared/dataset.json --out experiments/optuna_ctta_v1 --models cas_net --config-dir configs/plaque_workstation --ctta-config configs/ctta_mild.json --trials 12 --epochs 8 --steps 100 --timeout-hours 4
```

Use --models cas_net ffr_unet swin_unetr to search native models sequentially.
Each model has its own study. The timeout is per model and is checked between
trials; a currently running trial finishes before timeout enforcement. Re-running
the command adds --trials new trials to the existing compatible study. SQLite,
trial logs, requests and checkpoints are retained. Code/data/search-budget changes
are rejected; select a new --out for a changed study. A restarted sampler gets a
seed derived from the completed/running trial count to avoid repeating startup
random suggestions; interrupt/restart is not promised to match an uninterrupted
sampler sequence bit for bit.

Search space: learning rate 1e-5 to 1e-3, weight decay 1e-6 to 1e-3, plaque sampling
0.20-0.50, and mild step/step-motion probabilities 0-0.20 when CTTA is enabled.
Batch, accumulation and patch size remain fixed, so increasing training exposure
is not mistaken for a hyperparameter improvement. Validation patches remain fixed
and unaugmented. Default objective is validation PATCH plaque Dice; --metric
mean_dice uses the mean of lumen and plaque patch Dice instead. The test partition
is never used in the objective.

TPE sampling and a MedianPruner are used. Four trials finish before pruning is
enabled; at least three comparison trials are required at a step. Reporting is
zero-indexed, so n_warmup_steps=3 allows the first pruning check after epoch 4.
The child reports after saving a completed checkpoint and waits for an atomic
continue/prune acknowledgement. A pruned child exits normally, releasing its GPU
and worker resources before the next trial. Errors fail the search visibly with
the trial log path rather than silently treating software bugs as bad parameters.

Exports:
- studies.sqlite3: restartable per-model studies
- <model>/trial_XXXXX/trial.log and model/: per-trial evidence
- <model>_trials.csv and <model>_best.json: search summary
- best_config/<model>.json: settings for NEW full training, retaining the base
  25-epoch schedule rather than the short trial schedule

Example of training an exported native winner, with its own initialization:

```bat
"%GIT_BASH%" run_plaque_benchmark.sh init --bench experiments/plaque_all_accepted_physical --run-name native_tuned_ctta25 --a5000 --workstation --models cas_net --config-dir experiments/optuna_ctta_v1/best_config
"%GIT_BASH%" run_plaque_benchmark.sh train --bench experiments/plaque_all_accepted_physical --run-name native_tuned_ctta25 --a5000 --workstation --models cas_net --config-dir experiments/optuna_ctta_v1/best_config
```

Do not add --ctta-config here: the exported file already contains the selected
augmentation probabilities, and the CLI flag would replace them. Search winners
start fresh; trial weights are not promoted to final weights. Patch-Dice tuning
can overfit the fixed validation crops: confirm shortlisted settings using
full-volume validation and multiple seeds before evaluating the locked test set.
Report per-model search GPU-hours and sample budgets when comparing methods.

## Verification and limits

14 focused CPU tests passed: six real CTAug paths and deterministic sampling,
paired motion/ignore alignment, explicit insertion-policy validation, nnU-Net
normalization round trip, shared cache-key reuse with parallel building,
unaugmented validation, augmented resume/callbacks, real subprocess trials with
persisted study restart, and pruning after a saved checkpoint.
Two existing interrupted-resume and train/resume/infer/evaluate regression tests
also passed. Generated nnU-Net trainer syntax and the CTTA training hook were
exercised with a CPU contract stub against the inspected nnU-Net v2.8 API; validation
hooks and batch=2 were preserved. The full external nnU-Net/CUDA run has NOT been
executed in this workspace. Windows worker execution, augmented-model throughput
and scientific benefit require the local init/comparison runs. No speedup beyond
the measured unaugmented re-init results is claimed.

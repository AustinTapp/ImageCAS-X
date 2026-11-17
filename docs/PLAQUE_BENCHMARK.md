# SHK lumen and plaque benchmark

The extension adds SHK auditing, locked patient splits, multiclass training, native probability export, branch centerlines, sMPR reconstruction, inverse projection, evaluation, and fusion to ImageCAS-X. It reuses the existing model registry, architectures, and sliding-window inference.

This reissue rebuilds the source from the implementation in this conversation. The current synthetic suites were rerun. Read [the implementation status](PLAQUE_IMPLEMENTATION_STATUS.md) for verification scope and remaining gaps.

## Installation

Activate the configured PyTorch environment and run the following from the repository root.

```bat
python -m pip install -e ".[plaque]"
python -m pip install pytest
```

DynUNet uses optional MONAI. Select a release compatible with the installed PyTorch version.

```bat
python -m pip install --no-deps monai
```

Keep nnU-Net in the existing external environment. Install the benchmark there with `python -m pip install --no-deps -e .` and ensure nibabel and surface-distance are available. VMTK can run in a separate Python environment with NumPy, SciPy, nibabel, scikit-image, SimpleITK, VTK, and VMTK.

Set `PLAQUE_PYTHON` and `VMTK_PYTHON` to their executable paths when these environments differ. Batch launchers run from the repository root. The supplied defaults use the SHK image and annotation paths from the request.

## Audit, split, and baseline

```bat
call scripts\plaque_00_audit.bat
call scripts\plaque_01_prepare.bat
python -m plaque_benchmark train --model ffr_unet --profile smoke --out experiments\plaque_benchmark\runs\ffr_smoke
python -m plaque_benchmark infer --checkpoint experiments\plaque_benchmark\runs\ffr_smoke\best.pt --out experiments\plaque_benchmark\predictions\ffr_smoke_val
python -m plaque_benchmark evaluate --predictions experiments\plaque_benchmark\predictions\ffr_smoke_val --out experiments\plaque_benchmark\reports\ffr_smoke_val
```

CT filenames or parent directories must contain their SHK identifiers. Multiple CT candidates require review. The audit writes JSON and CSV with dimensions, affine, qform/sform, origin, spacing, direction, orientation, units, source hashes, labels, empty or missing branches, overlap, connectivity, plaque burden, and HU summaries.

Geometry mismatch, conflicting labels, and missing lumen exclude a case. Disconnected anatomy and overlapping branches require review. Plaque-free annotated branches remain eligible. Original files are read-only.

Header agreement leaves anatomical alignment and annotation completeness pending visual review. To require reviewed patients, provide CSV columns `case,status` with `accept` or `exclude` decisions.

```bat
python -m plaque_benchmark audit --review reviewed_cases.csv --out experiments\reviewed\audit.json
python -m plaque_benchmark split --audit experiments\reviewed\audit.json --require-reviewed --out experiments\reviewed\split.json
python -m plaque_benchmark prepare --audit experiments\reviewed\audit.json --split experiments\reviewed\split.json --out experiments\reviewed\prepared
```

Review decisions preserve hard geometry and label errors. Use a new experiment directory when changing an audit or split. Locked metadata rejects conflicting replacements. Source hashes bind prepared labels to the audited files.

The split balances patient plaque presence and affected branches with seed 2028. A cohort of 110 eligible patients yields 77 training, 16 validation, and 17 test patients. All branches of each patient stay together.

## Screening, targets, and sampling

```bat
call scripts\plaque_05_screen.bat
```

CAS-Net, FFR-UNet, and Swin UNETR run sequentially. Screening defaults to 25 epochs and 100 optimizer steps per epoch, 64³ patches, batch size 1, and accumulation 2. Smoke defaults to five epochs and two steps per epoch. Focused defaults to 80 epochs. The CUDA allocator cap defaults to 22 GiB, and supported devices use bfloat16. A5000 throughput and memory remain unmeasured.

Supported targets are `multiclass`, `lumen`, `plaque`, and `branch_lumen`. The last target predicts background and LAD/LCX/RCA lumen, with ambiguous overlapping lumen voxels ignored as 255.

Losses include Dice+CE, focal, focal Tversky, weighted CE, morphological boundary Dice, and lumen clDice. Sampling includes random, foreground, plaque, healthy branches, and training false positives.

Override profile settings using `--config overrides.json`. Nested dictionaries such as `sampling` are replaced as a unit. Best checkpoint selection uses deterministic validation patches. Full-volume inference and evaluation provide the comparison metrics.

```bat
python -m plaque_benchmark train --model ffr_unet --task plaque --out experiments\plaque_benchmark\runs\plaque_specialist
python -m plaque_benchmark train --model ffr_unet --task branch_lumen --config plaque_benchmark\configs\branch_lumen.json --out experiments\plaque_benchmark\runs\branch_lumen
python -m plaque_benchmark train --model ffr_unet --profile smoke --out experiments\plaque_benchmark\runs\ffr_smoke --resume
python -m plaque_benchmark train --model ffr_unet --profile focused --initialize experiments\plaque_benchmark\runs\ffr_smoke\best.pt --out experiments\plaque_benchmark\runs\ffr_focused
```

Resume restores the optimizer, schedule, completed epoch, and selection history. Initialization starts a new optimizer and schedule from compatible model weights.

For hard-negative mining, infer the training partition first, then provide `hard_negative_predictions` in the training configuration and a `sampling.hard_negative` fraction. Validation patches use independent sampling.

## Centerlines

```bat
call scripts\plaque_02_centerlines.bat
call scripts\plaque_03_smpr.bat
```

The adapter includes the supplied `vmtk_aorta_guided_core.py` verbatim. It reuses Flying Edges, physical transforms, aorta-interface localization, endpoint stabilization, VMTK extraction, and markups.

Use `--aorta-dir` and `--aorta-label` for matching aorta masks. Large ostium gaps reject automatic assignment. LAD and LCX are processed independently.

Without an aorta, skeleton topology and local trajectories supply candidate endpoints. Root identification uses a configurable caliber ratio, defaulting to 1.20. Ambiguity and disconnected masks produce review failures. Final centerlines come from VMTK.

For manual oracle roots, supply a JSON mapping from patient and branch to three RAS-mm coordinates.

```json
{"SHK0001": {"LAD": [12.0, -30.0, 85.0]}}
```

Use `--seeds roots_ras.json`. These runs carry a manual-supervision marker. The `--proxy` option uses lumen plus plaque as a vessel-course aid and records this separately.

Small surfaces retain full density when decimation would leave fewer than 1,000 points. Endpoint reach and path span reject truncated paths. QC also records radii, length, inside fraction, boundary distance, jumps, near self-approaches, duplicate paths, and orientation provenance.

Accepted and original VTPs, resampled VTP/NPZ paths, markups, and QC JSON are exported. VTP and markups use LPS mm. NPZ coordinates named `points_ras_mm` use RAS mm.

## sMPR and inverse mapping

Each accepted root-to-outlet path is reconstructed separately. Uniform arc-length sampling and parallel-transport frames define the local coordinates. CT uses linear interpolation and labels use nearest neighbors. Dense native-coordinate mappings and validity masks are saved. Reverse-folded samples are masked.

Centerline and reconstruction caching use input and implementation signatures. Shared trunks can occur in several paths and are averaged during native projection.

`--cross-mm` sets transverse sampling, `--fov-mm` requests a transverse extent, and `--cross-pixels` specifies a pixel count directly. `--long-axis` controls longitudinal axis placement. Longitudinal spacing comes from centerline resampling.

Inverse projection scatters plaque probabilities before thresholding and saves native coverage. Unrepresented native voxels remain unsampled. Missing score files are recorded as extraction/scoring failures.

## Moosavi pretrained inference

The earlier session inspected public inputs with 44 × 44 cross-sections, identity affines, and stored unit spacing. Released plans use a 3D PlainConvUNet, 48³ patches, and CT normalization. Source reconstruction deformation and physical equivalence remain unavailable.

Published label 1 denotes non-pathological vessel. SHK label 1 denotes patent lumen. The released model contributes plaque probabilities to the benchmark.

The released training foreground mean is approximately 1218, with clipping percentiles near 879 and 1875. The transfer intensity offset remains unresolved. The launcher requires an explicit `MOOSAVI_OFFSET`, and a candidate offset of 1024 requires validation against the actual native CT encoding.

```bat
python -m pip install gdown
python -m plaque_benchmark moosavi-fetch --out public_assets
set "MOOSAVI_MODEL=public_assets\Dataset001_CCTA_DHM\nnUNetTrainer__nnUNetPlans__3d_fullres"
python -m plaque_benchmark moosavi-inspect --model "%MOOSAVI_MODEL%" --samples public_assets\Sample_Dataset\Images --out public_assets\audit.json
python -m plaque_benchmark moosavi-infer --model "%MOOSAVI_MODEL%" --images public_assets\Sample_Dataset\Images --out public_assets\predictions --device cuda
python -m plaque_benchmark moosavi-verify --model "%MOOSAVI_MODEL%" --samples public_assets\Sample_Dataset --scores public_assets\predictions --out public_assets\verification.json
```

The downloader fetches the previously inspected fold-0 checkpoint, plans, metadata, and three public P01 vessels. Other folds are linked from the authors' repository.

For exploratory dimension matching, use `--cross-mm 1 --cross-pixels 44 --long-axis 2` and the desired longitudinal centerline step. Geometry equivalence remains unverified even with matching dimensions.

After selecting the intensity offset, `scripts\plaque_04_pretrained.bat` performs inference, inverse projection, and evaluation on oracle reconstructions.

## Custom sMPR models and fine-tuning

Generate training and validation centerline/sMPR indices independently, with labels included for training. Preserve healthy branches.

```bat
python -m plaque_benchmark smpr-dataset --indices experiments\smpr_train\smpr.json experiments\smpr_val\smpr.json --out experiments\smpr_training
python -m plaque_benchmark train --dataset experiments\smpr_training\dataset.json --model dynunet --task plaque --config plaque_benchmark\configs\highres_smpr.json --out experiments\smpr_training\runs\plaque
```

Native model output on reconstructed volumes uses CXYZ probabilities. Project it using `inverse --engine xyz --plaque-channel 1 --initialization random`. The Moosavi adapter uses `--engine nnunet`, which reads SimpleITK-order CZYX archives.

For fine-tuning in external nnU-Net, export the reconstructed data, retain compatible model plans, install the patient split, and initialize from released weights.

```bat
python -m plaque_benchmark nnunet-export --dataset experiments\smpr_training\dataset.json --raw "%nnUNet_raw%" --name Dataset703_SHKSMPR --intensity-offset "%MOOSAVI_OFFSET%"
python -m plaque_benchmark nnunet-transfer-plans --model "%MOOSAVI_MODEL%" --exported "%nnUNet_raw%\Dataset703_SHKSMPR" --preprocessed "%nnUNet_preprocessed%"
nnUNetv2_extract_fingerprint -d 703 --verify_dataset_integrity
nnUNetv2_preprocess -d 703 -plans_name MoosaviTransfer -c 3d_fullres -np 8
python -m plaque_benchmark nnunet-split --exported "%nnUNet_raw%\Dataset703_SHKSMPR" --preprocessed "%nnUNet_preprocessed%"
nnUNetv2_train 703 3d_fullres 0 -tr nnUNetTrainerSHK25 -p MoosaviTransfer -pretrained_weights "%MOOSAVI_MODEL%\fold_0\checkpoint_final.pth"
```

Inference inputs must use the same chosen intensity encoding. Standardize external probabilities with `nnunet-import` on the reconstructed dataset, then inverse-project with `--engine xyz`. Clinical fine-tuning remains pending.

## Automatic geometry

```bat
python -m plaque_benchmark infer --checkpoint experiments\plaque_benchmark\runs\branch_lumen\best.pt --out experiments\plaque_benchmark\predictions\branch_lumen_val
python -m plaque_benchmark centerlines --source predicted --predictions experiments\plaque_benchmark\predictions\branch_lumen_val --out experiments\plaque_benchmark\centerlines\predicted_val
python -m plaque_benchmark smpr --centerlines experiments\plaque_benchmark\centerlines\predicted_val\centerlines.json --out experiments\plaque_benchmark\smpr\predicted_val
```

This condition reads predicted branch-lumen masks for geometry. Manual seeds and plaque-derived proxies are rejected. External aorta inputs carry an upstream-provenance marker until their automatic origin is established.

## External native nnU-Net

Activate the configured nnU-Net environment with its existing raw, preprocessed, and result paths.

```bat
call scripts\plaque_nnunet_prepare.bat
call scripts\plaque_nnunet_screen.bat
```

Dataset 701 contains training and validation patients with a locked split. Protected test images are withheld from training export. The launcher targets 20 GiB for standard and residual-encoder plans and installs the split before training.

Reduced trainers provide five, 25, or 80 epochs with 100 optimizer steps per epoch. Planner-selected batch sizes require accounting for different patch counts.

```bat
python -m plaque_benchmark nnunet-inputs --out experiments\plaque_benchmark\nnunet_val_inputs
nnUNetv2_predict -d 701 -i experiments\plaque_benchmark\nnunet_val_inputs -o experiments\plaque_benchmark\nnunet_scores -c 3d_fullres -f 0 -tr nnUNetTrainerSHK25 -p nnUNetPlans_20G --save_probabilities
python -m plaque_benchmark nnunet-import --predictions experiments\plaque_benchmark\nnunet_scores --out experiments\plaque_benchmark\predictions\nnunet_val --checkpoint "%nnUNet_results%\Dataset701_SHKPlaque\nnUNetTrainerSHK25__nnUNetPlans_20G__3d_fullres\fold_0\checkpoint_best.pth" --initialization random
```

## Metrics, fusion, and reporting

Native predictions include a label NIfTI, CXYZ probability/coverage archive, and run provenance. Affine, dimensions, probabilities, coverage, split, and partition are validated.

Overlap, volume errors, surfel-weighted surface distances, HD95, surface Dice, and component detection are reported separately for lumen and plaque. Empty-GT overlap is null. Healthy branches have separate specificity, FP volume, FP lesion, and abstention summaries.

Lesions use 26-connectivity and one-to-one maximum-cardinality matching followed by IoU preference. Split/merge ambiguities are recorded. Native branch assignment uses reference branch proximity for evaluation only.

End-to-end metrics include missing plaque from failed extraction. Covered-only metrics exclude unsampled voxels. Specificity is conditioned on successful processing, with abstentions and successful-negative yield reported separately.

Use `geometry-metrics` for evaluation-only arc position, lesion extent, radial distance, lumen distance, and angular localization. The report identifies its geometry supervision.

Fusion supports voting, soft or validation-weighted averaging, confidence weighting, rule gating, specialist fusion, and logistic gating. All common voxel fusion occurs in native coordinates. Oracle inputs propagate oracle-assisted provenance.

`fit-gate` divides validation patients into calibration and assessment subsets. Gated evaluation excludes calibration cases. Apply `evaluate --assessment-from gate.json` to other comparators. Comparisons require identical assessment cases.

Postprocessing supports physical component volume, thresholding, independently predicted lumen distance, and explicit HU bounds. Defaults preserve small plaques. Probabilities are retained while label maps contain filtering.

`compare` generates CSV/JSON, Dice and healthy-FP plots, PR operating points, GPU-hour comparisons where measured, lesion-size sensitivity, and a case failure matrix. Training curves and sMPR overlays are separate products.

## Protected test access

```bat
python -m plaque_benchmark freeze --checkpoints experiments\plaque_benchmark\runs\ffr_smoke\best.pt --decision "Promote after validation review." --out experiments\plaque_benchmark\promotion.json
call scripts\plaque_08_test.bat experiments\plaque_benchmark\runs\ffr_smoke\best.pt experiments\plaque_benchmark\promotion.json
```

Promotion records bind the patient split, model identity, and evaluation protocol. For ensembles and reconstructed pipelines, promote their validation run identities with `freeze --runs` and promote constituent checkpoints before test inference. Keep normalization, reconstruction, and threshold choices fixed.

## Tests and central runner

```bat
call scripts\plaque_smoke_tests.bat
python -m plaque_benchmark synthetic --out experiments\synthetic_shk
```

The synthetic cohort contains nine artificial patients and RAS root seeds. Its directories can be used with the ordinary CLI audit, preparation, and geometry stages.

The central runner reads a JSON list of argument lists and executes stages sequentially, stopping on failure.

```json
[
  ["audit"],
  ["split"],
  ["prepare"],
  ["train", "--model", "ffr_unet", "--profile", "smoke", "--out", "experiments/plaque_benchmark/runs/ffr_smoke"]
]
```

```bat
python -m plaque_benchmark run --spec experiment_stages.json
```

## Sources

- [ImageCAS-X](https://github.com/AustinTapp/ImageCAS-X)
- [Moosavi repository and released assets](https://github.com/MM-DHM/nnUNet-Coronary-CTA-Segmentation)
- [Moosavi publication](https://doi.org/10.1371/journal.pone.0343887)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

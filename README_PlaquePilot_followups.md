# PlaquePilot: follow-up experiments and case review

Extract **the contents of this ZIP directly into `D:\Code\ImageCAS-X`**, beside your existing `run_PlaquePilot.py` and `pyproject.toml`. These are additional files; your benchmark package, dataset, existing checkpoints and launcher stay intact.

## 1. Install the viewer dependencies

In Anaconda Prompt:

```bat
conda activate PlaquePilot
cd /d D:\Code\ImageCAS-X
python -m pip install -r requirements_review.txt
```

The experiment launcher uses the benchmark dependencies already installed in PlaquePilot. The viewer requires no PyTorch or GPU. A synthetic demo is available:

```bat
python review_plaque_cases.py --demo --out experiments/PlaquePilot_viewer_demo
```

## 2. Run the next experiments

The recommended complete sequence is:

```bat
conda activate PlaquePilot
cd /d D:\Code\ImageCAS-X
call run_PlaquePilot_followups.bat all
```

Or paste this single line from any Anaconda Prompt:

```bat
conda activate PlaquePilot && cd /d D:\Code\ImageCAS-X && call run_PlaquePilot_followups.bat all
```

It runs two stages, in order:

1. **Validation-only probability experiments on CPU.** Three original argmax baselines plus 30 variants: plaque probability thresholds 0.35 / 0.50 / 0.65 crossed with component minima 0 / 1 mm³, applied to standard nnU-Net, ResEnc, FFR-UNet, an equal-weight nnU-Net+ResEnc ensemble, and an equal-weight nnU-Net+FFR ensemble. All ensemble predictions use the intersection of source coverage. Components use 26-connectivity and physical voxel volume. Lumen is not reevaluated in this plaque-only sweep.
2. **Paired CTTA-off training on GPU.** Standard nnU-Net, ResEnc, then FFR-UNet each undergo the existing one-epoch initialization gate followed by an 80-epoch maximum training run, with 100 optimizer steps per epoch. The source model's training settings are checked against the new initialization context before long training. Native patience, patch size, effective batch, optimizer and learning rate remain those of the actual source run. nnU-Net retains its existing plans and batch. Only the additional CTTA policy is removed; ordinary training augmentation remains enabled. FFR settings are copied from its actual trained override, including Optuna-selected values.

Defaults identify the runs used in your comparison:

```text
experiments/PlaquePilot_confirmed/runner/PlaquePilot_away80_tuned_ctta25
experiments/PlaquePilot_confirmed/runner/PlaquePilot_nnfix80_ctta25
```

Both nnU-Nets are included. CAS-Net and Swin are available as additional ablations but are not repeated by default because the current results give lower priority to further expensive runs of them:

```bat
call run_PlaquePilot_followups.bat all --ablation-models cas_net ffr_unet swin_unetr nnunet nnunet_resenc
```

For a shorter initial job, run only the CPU tests:

```bat
call run_PlaquePilot_followups.bat metrics
```

Run only the training ablation:

```bat
call run_PlaquePilot_followups.bat ablation
```

The CPU stage saves a per-patient cache. Unchanged completed ablation stages are reused; failed stages can be retried with the same command. Training uses the existing launcher's resume and initialization checks. A failed model is recorded and the remaining ablation models are attempted. Dataset preparation continues to use the existing confirmed cohort and locked patient split.

Output locations:

```text
experiments/PlaquePilot_followups_v1/validation_candidates.csv
experiments/PlaquePilot_followups_v1/validation_case_metrics.csv
experiments/PlaquePilot_followups_v1/provisional_winner.json
experiments/PlaquePilot_followups_v1/provisional_predictions/
experiments/PlaquePilot_followups_v1/ablation_status.json
experiments/PlaquePilot_followups_v1/ablation_logs/
experiments/PlaquePilot_followups_v1/ctta_paired_comparison/
experiments/PlaquePilot_confirmed/runner/PlaquePilot_CTTAoff_paired80_baseline25/
```

The `_baseline25` suffix is a legacy name: this launcher requests **80**, not 25, epochs. Native early stopping can finish earlier. It keeps Windows idle sleep inhibited while running; the display can turn off normally.

### How to interpret the CPU results

`validation_candidates.csv` gives positive-patient plaque Dice, macro precision/recall, healthy-branch FP rate, FP components per healthy branch, component detection sensitivity, sensitivity for components under 5 mm³, and paired patient-bootstrap Dice differences versus original standard nnU-Net. The bootstrap uses 5,000 draws and excludes patients with undefined plaque Dice; these intervals are exploratory and unadjusted for testing/selection across 33 candidates. Healthy-branch results retain their own denominators. Ground-truth connected components are a computational lesion definition.

The highest mean Dice candidate is exported **provisionally for review**. It is selected and assessed on the same validation patients, so this is not an unbiased confirmation of improvement. Check false positives and small-component sensitivity before choosing a candidate. Empty-reference plaque cases keep Dice undefined and still contribute to FP evaluation. Empty reference branches remain excluded from branch-specific scoring, consistent with the current evaluator.

Only the provisional winner's label NIfTIs are exported by default to limit storage. Its `run.json` has `label_only: true`; use it with the viewer. To create the large NPZ probability files required by the benchmark's full evaluator, use a distinct output directory:

```bat
call run_PlaquePilot_followups.bat metrics --export-probabilities --out experiments/PlaquePilot_followups_probs_v1
```

The CPU sweep omits surface-distance metrics. Existing full benchmark evaluations remain authoritative for those. No test patients are accessed by the follow-up launcher. After development, freeze the model and postprocessing choices before one held-out evaluation.

### Different run names or budgets

If your actual run names differ:

```bat
call run_PlaquePilot_followups.bat all --runs YOUR_NATIVE_RUN YOUR_NNUNET_RUN
```

If original training used a different maximum budget, match it with `--epochs` and `--steps`. The paired ablation deliberately stops before training when settings do not match; inspect `ablation_logs/*_configuration_mismatch.json` for exact values. Changing inputs/settings after a run starts requires a new `--out`; changing training configuration also requires a new `--ablation-prefix`.

If a model has multiple prediction fingerprints, the scripts require an explicit selection instead of choosing one by modification time. For the CPU sweep:

```bat
python run_PlaquePilot_followups.py metrics --prediction nnunet="FULL_PREDICTION_DIRECTORY" --prediction nnunet_resenc="FULL_PREDICTION_DIRECTORY" --prediction ffr_unet="FULL_PREDICTION_DIRECTORY"
```

Each directory must contain its `run.json`, case `.nii.gz` files and probability `.npz` files. Explicit directories alone are insufficient for the paired training ablation, which also needs the original standard runner layout and training configuration.

## 3. Open the interactive reviewer

In another Anaconda Prompt, or after the experiments finish:

```bat
conda activate PlaquePilot
cd /d D:\Code\ImageCAS-X
python review_plaque_cases.py
```

The default input is `experiments/PlaquePilot_confirmed/prepared/dataset.json`, validation partition, with all five prediction sets from the two runs above. The summary ZIP alone does not contain the CTs, masks or probabilities; the reviewer reads those from your local benchmark folders.

The left column displays ground truth over CT; the right column displays the selected model over the **same CT array**. Axial, coronal and sagittal views are visible together. Sliders and wheel scrolling operate in tandem, and clicking either side updates the crosshair in all six views. Window/level and overlay opacity are shared. Zoom and pan are linked within each left/right pair. Switching models on the same case reuses the loaded CT/ground truth and retains the slice position.

The case table shows recorded per-case plaque Dice when a source report is available. The header identifies recorded full-volume metrics versus overlap calculated on the display grid; display resampling can change the latter. Click **Sort by plaque Dice ↑** to start with poor overlaps. Empty-reference Dice is shown as undefined, rather than assigned a perfect score. Missing predictions remain in the case list; you can flag or reject them, but approval requires a successful load.

| Action | Key / control |
|---|---|
| Approve current case + model | **A** |
| Reject current case + model | **R** |
| Flag for further review | **F** |
| Reset decision to unreviewed | **U** |
| Next / previous case | **N / B**, or **Right / Left** |
| Toggle ground truth overlay | **G** |
| Toggle prediction overlay | **P** |
| Toggle both overlays | **Space** |
| Jump to an annotated plaque voxel, or predicted plaque if absent | **J** |
| Fit the six views | **Home** |
| Scroll slices | Wheel over either image, or paired slider |
| Step the last active plane | **Up / Down** |
| Zoom | **Ctrl + wheel**, or right drag |
| Pan | Middle drag |

Hotkeys are suspended while typing notes or editing numeric controls. The GT and prediction checkboxes also toggle overlays independently. Lumen is cyan, plaque orange, ignored label 255 magenta. Individual label checkboxes and opacity controls support unobstructed CT inspection. Orientation labels use radiological convention: patient right appears on the left of axial/coronal views.

### Decisions and case folders

Decisions apply to a **case + prediction set**, so approving one network does not approve its other predictions. The default review root is:

```text
experiments/PlaquePilot_case_review/
```

It contains `approved`, `rejected`, `needs_review` and `unreviewed` folders, subdivided by prediction set and case. Each reviewed case package contains CT, combined ground-truth mask, prediction mask if available, and `review.json` with source paths, review notes, reason, reviewer, timestamp and view state. `reviews.csv` is the spreadsheet view, `reviews.json` the authoritative current index, and `history.jsonl` the decision history. Reclassification transfers the review package into its new category. Missing input files are recorded explicitly.

**Original dataset files, model outputs, dataset acceptance and train/validation/test membership stay unchanged.** A rejection here means that prediction needs attention; it does not remove an accepted case from future training.

To avoid copying the same CT five times, the script first copies originals into a review-owned asset cache, then uses local hard links into category packages when supported. Windows NTFS normally supports these without administrator privileges. It falls back to copying when hard links are unavailable. No links are made to the original files. Editing a review copy can affect other review packages sharing that cached asset, so use the review collection as read-only image evidence. The first decision for a large CT can take a moment; copying occurs in the background and the interface shows its status. A later decision on the same case usually reuses its cached assets.

For metadata-only category folders, use:

```bat
python review_plaque_cases.py --references-only --out experiments/PlaquePilot_review_references
```

### Geometry

CT and overlays are aligned by their NIfTI physical affines. Canonical orientation is RAS. Oblique CT is resampled onto an orthogonal RAS grid for display, using linear CT interpolation and nearest-neighbor label interpolation; original files are preserved. The same display CT is used on both sides. Conflicting qform/sform definitions raise a visible error.

A label grid that differs from the original CT requires explicitly enabling **Resample mismatched labels for display**, or starting with `--resample-labels`. This uses the existing physical affine and does not establish registration correctness. Actual alignment errors remain review findings. No case is silently dropped for a geometry failure; use Further review and add a reason.

### Review the provisional CPU candidate alongside a reference model

Replace the reference path with the unique standard nnU-Net prediction fingerprint:

```bat
python review_plaque_cases.py --prediction provisional=experiments/PlaquePilot_followups_v1/provisional_predictions --prediction nnunet="experiments/PlaquePilot_confirmed/runner/PlaquePilot_nnfix80_ctta25/predictions/nnunet/YOUR_CHECKPOINT_FINGERPRINT"
```

Or review only the exported candidate:

```bat
python review_plaque_cases.py --prediction provisional=experiments/PlaquePilot_followups_v1/provisional_predictions
```

Other cohorts can be opened with `--dataset PATH --prediction NAME=DIRECTORY`; partition defaults to `val`. The viewer can display already-generated `train`/`test` sets when explicitly requested with `--partition`, but using test ground truth for development would invalidate an untouched test assessment.

## 4. Verification

Synthetic checks cover physical reorientation, explicit mismatched-grid resampling, oblique CT display, qform/sform conflicts, six-panel rendering and tandem scrolling, overlay hotkeys, model switching, preserved originals, reclassification and failed exports. The complete CPU sweep, provisional NIfTI export and restart were exercised on two synthetic patients, including an empty-plaque reference. Paired ablation command construction, configuration matching and completed-stage reuse were checked using simulated training processes.

Real Windows GUI rendering, real cohort throughput and CUDA/nnU-Net training need your workstation run. Existing initialization gates are retained for that purpose. Run tests locally, if desired:

```bat
python -m pip install pytest
python -m pytest -q tests/test_review_and_followups.py
```

# PlaquePilot execution launcher

Inspected public branch: AustinTapp/ImageCAS-X, PlaquePilot,
commit c597d3334d957f987502a1a0f612738ae7e4a1fc.

## Installation review

The pushed branch contains pyproject.toml, the full plaque_benchmark package and
its JSON profiles, run_plaque_benchmark.sh, the workstation/CTTA/Optuna modules,
requirements_workstation.txt, and the CTTA JSON policies. The package-discovery
configuration includes plaque_benchmark and packages its internal JSON profiles.
PyTorch is intentionally installed separately. MONAI is not declared in the
core/optional requirements, so an installation cloned from omni should retain it;
the launcher checks that it is actually importable. If absent, install the
previously used version with: python -m pip install monai==1.5.2.

The local nnunet dependency directory is Git-ignored. Its files and your Windows
Python environment cannot be inspected from GitHub. The launcher therefore checks
these locally, before data preparation: active interpreter/prefix, CUDA/GPU,
Torch/MONAI/CTAug/Optuna imports and versions, loaded native source path, nnU-Net
console entry points, writable trainer directory and training-transform API.
If the nnunet folder exists, the active nnunetv2 import must resolve inside it.
A stale editable import from the former omni/CAS checkout produces an actionable
error. The same PlaquePilot Python is used for native and external commands;
stale PLAQUE_PYTHON or NNUNET_PYTHON shell settings do not choose another interpreter.

The existing *.json ignore rule can hide future NEW source configurations, although
the required configurations are already tracked. Do not commit experiments or
nnU-Net data/results. This archive contains only execution code, documentation,
tests and a small evaluation correction.

## Install these files

Stop active training, then extract this ZIP directly into D:\Code\ImageCAS-X.
It adds run_PlaquePilot.bat, run_PlaquePilot.py, this document, and regression tests.
It replaces plaque_benchmark/evaluation.py with the version from the inspected
branch plus empty-reference-branch handling. No images, masks or checkpoints are
in the archive. The code change requires fresh initialization receipts, as intended.
No additional accepted-cases helper download is required.

Use Anaconda Prompt. The BAT wrapper activates PlaquePilot and uses this checkout.
It invokes the Python body of the existing trusted run_plaque_benchmark.sh, retaining
its epoch/checkpoint gates. Git Bash is not needed for these wrapper commands.

## Main sequence

```bat
cd /d D:\Code\ImageCAS-X
run_PlaquePilot.bat init
```

This command:
1. Checks the local installation and records local_installation.json.
2. Discovers every SHK case under D:\Data\PlaqueDataset\AsNifti and
   D:\Data\PlaqueDataset\MasksUnverified. Every discovered case is accepted on
   the basis of your explicit visual alignment confirmation in the conversation.
3. Creates the accepted mask copies, combined labels and approval record. The
   audit.csv review column is filled with accept automatically; do not edit it.
4. Reuses experiments/plaque_all_accepted_physical/split.json if present and if it
   covers exactly the same cases. Otherwise, when no previous split exists,
   creates the locked 70/15/15 patient split with seed 2028. The same new manifest
   is shared by the baseline, augmented and tuned experiments.
5. Allocates a FREE nnU-Net dataset ID in 710-799, without overwriting Dataset701
   or any occupied dataset ID. It honors the currently defined nnUNet_raw,
   nnUNet_preprocessed and nnUNet_results paths, or creates their ignored local
   directories under this checkout when they are unset. The chosen ID/paths are
   saved and reapplied on later runs.
6. Exports/preprocesses standard and residual nnU-Net plans when needed. Existing
   matching data/plans/splits are reused. Conflicting metadata stops execution.
7. Runs one initialization epoch and checkpoint verification for CAS-Net,
   FFR-UNet, Swin UNETR, nnU-Net and residual nnU-Net, sequentially.
8. Stops before long training so you can review the summary/logs.

Then, after initialization succeeds:

```bat
run_PlaquePilot.bat train
```

Training is sequential, with validation inference, evaluation and comparison.
It verifies every selected initialization receipt first. Existing compatible
long-training checkpoints are resumed automatically; unfinished models start
normally. It never resumes an old baseline with a changed augmentation policy.

Default output directory:

```text
D:\Code\ImageCAS-X\experiments\PlaquePilot_confirmed\
    approval_receipt.json
    audit.csv
    audit.json
    split.json
    local_installation.json
    nnunet_execution.json
    accepted_branches\
    prepared\dataset.json
    runner\PlaquePilot_ctta25\init_summary.csv
    runner\PlaquePilot_ctta25\init.log
    runner\PlaquePilot_ctta25\logs\
    runner\PlaquePilot_ctta25\training\
    runner\PlaquePilot_ctta25\comparison\
```

Rerunning init reuses the approved cohort, completed case preparation and matching
nnU-Net preprocessing. Unchanged successful initialization checks are reused by
the original runner. An interrupted preparation can resume already completed
cases. Source or derived-label changes after approval are detected using recorded
file identity/stat information and locked metadata. Initial source content hashes
are retained in the audit; the routine fast reuse check does not rehash every
large CT. Use a new --bench for changed inputs instead of editing sealed files.

If a prior split covers different cases, ALL cases remain selected and execution
stops before training. For an intentionally separate cohort, choose a new bench
and explicitly request a new split:

```bat
run_PlaquePilot.bat init --bench experiments/PlaquePilot_new_cohort --fresh-split
```

Repeat those options for every subsequent train/search command for that cohort.
This avoids silently moving prior test patients into training.

## Approval/geometry behavior

Your visual confirmation replaces the manual per-case approval task. It does not
require running the old strict audit/prepare commands. The generated audit records
basic empty/no-lumen, overlap and physical-grid findings; topology is not re-audited.

Original CTs and masks stay unchanged. Mask arrays are copied onto the CT grid
without voxel changes when grids match. If the grids differ, labels are resampled
using the ORIGINAL physical transform and nearest-neighbor interpolation. This
never uses the former voxel-index geometry reassignment. Unknown mask spatial
units are interpreted as mm; meter/micron units are converted explicitly. The
NIfTI-selected affine (sform precedence) is used when coded transforms disagree,
and that conflict is recorded. This is resampling, not image registration.

All accepted cases remain selected even for empty/no-lumen masks or branch overlap.
Plaque label 2 wins conflicts with lumen label 1 in the combined target; overlapping
branch-lumen IDs get ignore=255. Pairing ambiguity, missing/unreadable inputs,
nonfinite CTs, invalid physical transforms and unsupported numeric labels outside
0/1/2 stop the cohort as a whole. No subset manifest is trained after an input error.

The evaluation correction handles empty accepted reference branches. They remain
in CASE-level scoring. Empty branches are explicitly listed as unevaluable and
excluded from branch-specific attribution/specificity instead of being fabricated
as healthy branches. Other nonempty branch attribution and case-level metrics
are unchanged. An empty reference branch and a plaque-free nonempty branch are
different situations. Predictions are still scored at case level.

## Resources and experiments

Default native settings: CAS batch 32/accumulation 1, FFR 8/4 and Swin 4/8,
64-cube patches, 100 updates/epoch, up to 25 epochs. Four native loader workers,
four cache builders, four nnU-Net preprocessing workers and bounded prefetching.
The existing mild CTTA policy is used. nnU-Net retains its planned batch=2,
20G plans and 22 GiB allocator cap. Its measured ~21.2-21.6 GiB peaks do not justify
increasing the batch. Original native memory-mapped caching/tqdm remain active.

One coordinator runs at a time per checkout using an OS file lock. Do not launch
the old shell launcher concurrently against the same GPU or cohort.

Additional commands:

```bat
rem Installation report only; no dataset approval or training
run_PlaquePilot.bat doctor

rem Approval and label/split preparation only; no nnU-Net preprocessing or training
run_PlaquePilot.bat prepare

rem Optional matched non-CTTA baseline using the SAME new cohort/split
run_PlaquePilot.bat baseline-init
run_PlaquePilot.bat baseline-train

rem Native-only Optuna search; default 12 additional trials/model, 8 epochs,
rem four-hour timeout/model checked between trials. Start with CAS-Net.
run_PlaquePilot.bat search --models cas_net

rem Initialize and train exported native winners, under a separate run name
run_PlaquePilot.bat tuned-init
run_PlaquePilot.bat tuned-train
```

Search defaults to the three native models when no --models option is given.
It uses the existing Optuna search implementation and never tunes on held-out test
patients. Search completion is separate from long training. Tuned stages select
available exported native configurations and use the exported augmentation settings;
they do not overwrite them with the default mild policy. External nnU-Net HPO is
not implemented. A matched baseline is optional; the default init/train commands
run the augmented five-network experiment only.

Examples for a smaller initialization check:

```bat
run_PlaquePilot.bat init --models cas_net ffr_unet swin_unetr
```

Reuse the same --models selection when training. To initialize/train the full
sequence later, run the default commands; model-specific checks remain gated.
Change --run-prefix to create a separate experiment group. Do not change learning
settings or the prepared cohort while resuming existing checkpoints.

## Verification and practical limits

The inspected Git structure is verified. Your Windows environment/import paths,
GPU visibility, disk space and actual installed nnU-Net dependencies are checked
locally by doctor/init; they were not remotely inspected on your computer.

Regression tests cover automatic acceptance without dropping flagged cases,
physical-coordinate resampling, original-file preservation, idempotent approval,
changed-input rejection, patient split reuse/mismatch rejection, unsupported-label
failure, recovery of interrupted case preparation, free-ID allocation and persisted
nnU-Net paths, preprocessing reuse/conflict guards, and explicit empty-branch
metrics. A real small CPU run executes the embedded runner through initialization,
its checkpoint gate, sequential training, inference, evaluation and comparison,
and confirms initialization does NOT launch training. Windows BAT/Conda and the
external CUDA nnU-Net preprocessing/training still need your local init run.

Optional tests from the repository root:

```bat
python -m pytest -q tests/test_pilot_execution.py
```

If you want to track the launcher on GitHub after installing it:

```bat
git add run_PlaquePilot.bat run_PlaquePilot.py PLAQUEPILOT_EXECUTION.md plaque_benchmark/evaluation.py tests/test_pilot_execution.py
git commit -m "Add staged PlaquePilot execution and confirmed-cohort approval"
git push
```

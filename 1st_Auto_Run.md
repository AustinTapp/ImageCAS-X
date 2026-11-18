# PlaquePilot unattended five-model run

Extract this updated launcher archive directly into D:\Code\ImageCAS-X, replacing
matching files. The original execution helper and its empty-branch evaluation
correction are included, so a separate execution-launcher download is unnecessary.
The workstation/CTTA/Optuna codebase and installed PlaquePilot environment are
prerequisites. Stop active jobs before replacing code.

In Anaconda Prompt, paste:

```bat
cd /d D:\Code\ImageCAS-X && call run_PlaquePilot_unattended.bat
```

There are no review prompts between stages. This is explicit authorization to
proceed past initialization into longer training, as requested for the away run.

## Actual model coverage

| Model | Search | Longer training |
|---|---|---|
| CAS-Net | 4 native Optuna trials, up to 4 epochs each | Exported winner, up to 80 epochs |
| FFR-UNet | 4 native Optuna trials, up to 4 epochs each | Exported winner, up to 80 epochs |
| Swin UNETR | 4 native Optuna trials, up to 4 epochs each | Exported winner, up to 80 epochs |
| nnU-Net | Existing standard 20G plan; no Optuna search | 80 epochs |
| Residual-encoder nnU-Net | Existing residual 20G plan; no Optuna search | 80 epochs |

All five methods use the mild CTTA policy and the SAME approved cohort/patient
split. Native effective batch=32, 100 updates/epoch and the previous native
patience setting remain. Native runs may stop before 80 epochs on early stopping.
Both external models retain batch=2 and the 22 GiB allocator cap; their measured
peaks were already ~21.2-21.6 GiB. Optional DynUNet, Moosavi/sMPR, and incomplete
multistage methods are outside this five-model workflow.

Sequence:
1. Verify the PlaquePilot interpreter, imports, GPU and local nnU-Net installation.
2. Reuse or build the automatically approved physical-coordinate cohort. Your
   existing confirmation of MasksUnverified/AsNifti alignment is the approval basis.
3. Allocate/reuse a free nnU-Net dataset ID and prepare missing plans/data.
4. Initialize each of the five methods independently with the long-run settings.
5. Run LONG TRAINING for standard nnU-Net and residual nnU-Net, including validation
   inference/evaluation. This occurs before native search/long jobs.
6. For each native method, run its light search, verify its exported winner with
   initialization, then perform longer training and validation inference/evaluation.
7. Generate a combined comparison across completed models, with configurations
   labelled as planned nnU-Net, Optuna winner, or fixed-profile fallback.

This does NOT add nnU-Net hyperparameter search. It ensures both nnU-Net variants
are actually trained, rather than only initialized.

## Failure isolation and fallback

A per-model initialization/training error is recorded and does not prevent later
independent models from being attempted. If a native search fails to export a
usable winner, or winner initialization fails, the previously verified FIXED CTTA
configuration is trained instead. Fallbacks are explicit in the summary and plots;
they are never described as tuned winners. If that model's fixed initialization
also failed, it remains marked failed/untrained.

Invalid common dataset inputs, unavailable CUDA, unusable installations or
changed completed-stage artifacts stop the workflow rather than silently changing
the experiment. A shared nnU-Net preprocessing error is recorded; each nnU-Net
initialization then checks its own required plan, while native jobs can continue.

The search time budget defaults to one hour PER NATIVE MODEL, checked between
trials. A running trial may extend past it. Depending on that budget, fewer than
four trials may finish. The pruner needs four completed startup trials, so a light
four-trial search may complete without pruning; it is intentionally small.

## Resume without repeating all work

After an interruption or reboot, paste the SAME command again. Completed stages
are reused after checking their saved artifacts. Compatible incomplete longer
training resumes from its last completed epoch. Completed searches are not repeated.
If a search itself was interrupted, only the remaining trial-attempt budget is
requested; the interrupted trial counts toward that light budget. It is not resumed
mid-trial. If no usable winner remains after the budget, fixed-profile fallback
training preserves coverage. Once a model has completed fallback training, restarting
the same workflow does not launch a new search behind it.

Source/configuration/environment or cohort changes require a new --run-prefix
and --search-out. Restoring an accidentally changed completed artifact also allows
the original workflow to continue. Do not edit code, source images or masks while
the run is active. No existing baseline checkpoints are overwritten.

The legacy per-model output folder suffix still contains 'ctta25'; the actual
saved configuration, workflow settings and requested epoch count govern training.
The new default prefix is PlaquePilot_away80, and long-run configurations use 80.

## Progress and results

Default coordinator folder:

```text
experiments\PlaquePilot_confirmed\unattended\PlaquePilot_away80\
    console.log
    workflow.json
    model_summary.csv
    model_summary.json
    comparison\comparison.csv
    comparison\comparison.png
```

workflow.json is written at stage start and completion and records individual
failures. model_summary is written after model stages have been attempted. Per-model
training logs, tqdm output, histories and checkpoints remain in the normal runner
folders under experiments\PlaquePilot_confirmed\runner\PlaquePilot_away80_*.
Optuna studies/trial logs are under experiments\PlaquePilot_away_light_v1.

The combined comparison includes only successful model reports; model_summary.csv
explicitly lists failed/untrained models. Check model_coverage_complete and
failed_or_fallback_stages in workflow.json. No failed model is assigned an invented
score. Unaugmented matched-baseline runs remain a separate optional experiment.

## Windows idle sleep

While running, the coordinator calls Windows SetThreadExecutionState with
ES_CONTINUOUS|ES_SYSTEM_REQUIRED and clears its request on exit. This prevents
idle sleep while allowing the monitor to turn off. It does not change the saved
power plan and cannot prevent a manual sleep/shutdown, forced reboot, or power
failure. Keep the computer powered and the Anaconda Prompt open. It is a foreground
local job, not a service that survives closing the terminal.

## Optional controls

```bat
run_PlaquePilot_unattended.bat --epochs 100 --trials 6 --search-epochs 6 --search-hours 2 --run-prefix PlaquePilot_away100 --search-out experiments/PlaquePilot_away100_search
```

Do not reuse an existing prefix/search folder for changed settings. Existing
--images, --masks, --bench, --reuse-split and --fresh-split controls are available;
see --help. The global per-checkout execution lock prevents simultaneous execution
of this coordinator and the original staged wrapper. Run one GPU workflow at a time.

## Verification

Controller tests cover all five model options, actual nnU-Net long-training dispatch
before native search, isolation of an individual model failure, explicit fixed
fallback, reuse of completed fallback models, checkpoint/winner tamper rejection,
and changed-workflow rejection. A small real CPU run executes initialization,
Optuna subprocess search, exported-winner initialization, longer training,
inference/evaluation, combined comparison, and a second invocation that confirms
no extra search trials are launched. The staged execution launcher's nine prior
regression tests remain included. Actual multi-day Windows/CUDA/nnU-Net execution
cannot be validated on the CPU test host; individual failures are logged visibly.

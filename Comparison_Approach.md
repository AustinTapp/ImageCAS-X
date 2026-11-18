# Five-model comparison and next experiments

Source: the supplied `comparison_all5_nnfix80.zip` (`comparison.csv`, `comparison.json`, and comparison figures). Each model reports 16 assessment cases. From the existing pipeline these are validation results. The archive contains aggregate results and figures, not training logs, probability volumes, numerical per-case reports, or nnU-Net timing.

| Model | Plaque Dice | Lumen Dice | Plaque precision | Plaque recall | Healthy-branch FP rate | FP components / healthy branch |
|---|---:|---:|---:|---:|---:|---:|
| CAS-Net | 0.3800 | 0.5174 | 0.2623 | 0.4855 | 100.00% | 5.9375 |
| FFR-UNet | 0.4883 | 0.6497 | 0.3274 | **0.6161** | 56.25% | 2.6875 |
| Swin UNETR | 0.4348 | 0.4592 | 0.3166 | 0.5179 | 75.00% | 4.3750 |
| Standard nnU-Net | 0.5064 | 0.7159 | **0.4956** | 0.5263 | **12.50%** | **0.1875** |
| ResEnc nnU-Net | **0.5076** | **0.7447** | 0.4284 | 0.5189 | 18.75% | 0.2500 |

These are the reported macro averages. Positive-case Dice and healthy-branch false positives have different denominators. A healthy-branch FP rate is the fraction of annotated plaque-negative branches with any predicted plaque, not the voxel false-positive rate. FP lesion/component counts use the evaluator's 26-connected-component definition and nearest annotated branch assignment in physical space. That attribution is for evaluation, not inference. Unsupported empty reference branches are excluded by the current evaluator.

## What the outcome supports

1. **Keep both nnU-Nets as leading candidates.** The ResEnc plaque Dice advantage over standard nnU-Net is only 0.00126 (0.126 percentage points). Aggregate means do not establish a reliable ranking between them. ResEnc improves lumen Dice by 0.02883; standard nnU-Net has better plaque precision, recall and healthy-branch behavior at the reported operating point.
2. **Retain FFR-UNet as a recall-oriented comparator and ensemble candidate.** Its plaque recall is highest, but its false-positive burden is much higher. Standard nnU-Net's mean plaque Dice is only 0.01813 higher than FFR-UNet, while the healthy-branch FP rate drops from 56.25% to 12.5%. A recall-versus-FP comparison is more informative than Dice alone.
3. **Prioritize inspection of SHK0007**, which has very poor plaque overlap across all five models in the failure matrix. SHK0152 is another difficult case. Shared difficulty can reflect annotation uncertainty, lesion phenotype, small lesions or common preprocessing/model limitations; the figure alone cannot identify the cause. Preserve these cases in the accepted cohort and tag concerns during review.
4. **Inspect plaque-negative patients for false positives.** The white cells at SHK0001, SHK0015, SHK0073 and SHK0148 in the positive-case Dice matrix are undefined Dice entries, not proof of failed inference. In the current evaluator this occurs when ground-truth plaque is absent. These patients remain valuable for specificity/FP analysis.
5. **Small lesions remain a clear weakness.** The size-sensitivity figure shows the lowest sensitivity below 5 mm³, including for the nnU-Nets. Minimum-component filtering must therefore be checked against lost true small components, even when mean Dice improves.

The archive cannot show whether longer training is warranted, whether nnU-Net timing is favorable, or whether any score difference is statistically stable. Native GPU hours are present; nnU-Net GPU hours are missing and must not be interpreted as zero. A 16-patient validation set also limits model-selection certainty.

## Recommended experimental order

**First: bounded probability/postprocessing experiments, with no retraining.** Sweep thresholds 0.35 / 0.50 / 0.65 and small-component minima 0 / 1 mm³ for standard nnU-Net, ResEnc and FFR-UNet. Test equal-weight standard nnU-Net+ResEnc and standard nnU-Net+FFR ensembles. The purpose is to characterize the precision/recall/FP tradeoff and test complementarity; an ensemble is not presumed better. The supplied launcher performs this as 33 total candidates including the three original argmax baselines. Inspect patient-level Dice differences, healthy-branch FP rates, component sensitivity and sensitivity under 5 mm³ together. Its bootstrap intervals are exploratory, not a correction for searching 33 choices on the same patients.

**Second: paired CTTA ablation for the leading three models.** Compare the existing CTTA-on runs with CTTA-off runs retaining their actual architecture, patch, batch, learning rate, optimizer, maximum budget and ordinary augmentation. This determines whether the extra artifact augmentation helps these data. The provided launcher initializes each ablation first and checks that its configuration matches the original except CTTA before allowing long training. Defaults are nnU-Net, ResEnc and FFR-UNet. An all-five option is included.

**Third: repeat the best one or two configurations across at least three independent training seeds, or use patient-level cross-validation**, depending on the available compute budget. Add the reporting support for explicit seed/fold control before launching those runs; the supplied script does not disguise run-name changes as new random seeds. Consider longer budgets only after examining the winning models' training curves. For FFR, a larger physical field of view or training-only hard-negative mining would be focused next tests if review confirms background confusion. Ground-truth branch masks must not be used as inference gates.

**Finally: freeze the selected model, probability threshold and component policy, then perform one untouched test assessment.** Manual review of validation outputs is part of development. Keep prediction acceptance decisions separate from dataset inclusion. Do not remove difficult accepted cases to improve reported performance. If genuine annotation problems are corrected, version those labels and rerun comparisons consistently.

Optuna remains useful after the data/QC and evaluation choices are stable, especially for a bounded search on the leading architecture. Another broad search over every model is lower priority than the paired augmentation test, operating-point checks and seed/fold robustness. Very short trial objectives based on patch Dice can favor settings that do not improve full-volume performance.

See `README_PlaquePilot_followups.md` for the executable commands, viewer controls, output locations, restart behavior and limitations.

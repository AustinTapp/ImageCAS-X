# Engineering status for the reissued package

This package reconstructs the implementation from this conversation on ImageCAS-X commit `e4c571d0bd3fd2e2c997b36d13fefd220b8095b1`. Source organization and command interfaces follow the earlier delivery. Fresh verification of the reissue passed 27 native tests and four actual-VMTK tests. JUnit records are included under `docs/verification`.

## Models

| Model or condition | Available implementation | Verification and limits |
|---|---|---|
| CAS-Net | Configurable multiclass output head on existing architecture | Multiclass forward and default binary compatibility tests |
| FFR-UNet | Configurable head, training, initialization, resume and tiled inference | Fresh CPU training, checkpoint, interruption/resume and evaluation tests |
| Swin UNETR | Existing architecture with multiclass output | Reduced-width forward test, full-size CUDA training pending |
| Standard nnU-Net | External dataset export, locked split, reduced trainer, plans and probability import | Fresh interchange tests, earlier session executed one actual CPU training step |
| Residual-encoder nnU-Net | External planner and screening launcher | Local planning and training verification pending |
| Moosavi pretrained | Released assets, original nnU-Net preprocessing, sequential inference and probability export | Earlier session ran fold 0 on three public vessels, current reissue source compilation |
| Moosavi fine-tuning | Transferred plans, explicit intensity offset and pretrained initialization command | Complete clinical fine-tuning remains pending |
| DynUNet | Optional MONAI residual DynUNet adapter | Fresh reduced-filter forward test |
| Lumen and plaque specialists | Binary tasks and specialist fusion | Fresh preparation and fusion tests |
| Branch-lumen model | Background, LAD, LCX and RCA lumen training target | Fresh target tests, clinical model training pending |
| Lightweight branch model | Reduced FFR-UNet configuration | Configuration provided, branch-localization study pending |
| High-resolution sMPR model | DynUNet configuration and reconstructed-data manifest | Geometry and DynUNet tests, full training pending |
| ImageCAS multistage | Existing lumen implementation retained | Plaque-specific adaptation remains incomplete |
| ADE-HTL | Existing lumen implementation retained | Plaque-specific multitask adaptation remains incomplete |
| Additional hybrids | Primary Swin candidate is available | Further distinct candidates remain incomplete |

## Components

| Component | Implemented behavior | Verification |
|---|---|---|
| SHK discovery and audit | Geometry, labels, missing or empty branches, overlap, continuity, plaque-free examples and source hashes | Fresh deliberate-failure tests |
| Split and preparation | Patient balancing, immutable manifests, four target modes | Fresh tests, including 77/16/17 allocation |
| Native training | Fixed steps, accumulation, optional AMP, losses, sampling, checkpoints, initialization and early stopping | Fresh CPU tests |
| Hard negatives | Training prediction mining, independent validation sampling | Fresh sampling test |
| Aorta-guided VMTK | Supplied interface-centroid and endpoint stabilization logic | Fresh actual-VMTK test |
| Aorta-free VMTK | Endpoint topology, trajectory, root caliber or supplied seed, final VMTK extraction | Fresh ambiguity and VMTK tests |
| Centerline QC and export | Radius, physical frame, endpoints, target reach, path span, near self-approach, jumps, accepted/original/resampled outputs | Fresh actual-VMTK and narrow-vessel tests |
| sMPR | Physical interpolation, stabilized frames, categorical labels, mapping, validity and folding mask | Fresh correspondence and round-trip tests |
| Inverse projection | Probability scattering, overlap average, coverage and failure records | Fresh inverse and extraction-failure tests |
| Probability interchange | Grid, class, split and coverage validation | Fresh tests |
| Evaluation | Overlap, surface, lesions, branch and healthy-vessel summaries | Fresh tests |
| Geometric evaluation | Lumen distance, centerline extent, local radius and angle | Implemented, dedicated clinical validation pending |
| Postprocessing | Size, threshold, predicted lumen distance and HU bounds | Fresh synthetic test |
| Simple fusion | Voting, average, supplied weights, confidence and rule gating | Fresh array and full fusion tests |
| Learned gate | Held-out validation calibration, independent assessment | Fresh fit and exclusion test |
| Reporting | Tables, learning curves, PR operating points, negative FP rate, lesion-size plots and failure matrix | Fresh plotting test |
| Windows launchers | Audit through protected test plus nnU-Net support | Source restored, Windows execution pending |
| Protected test | Frozen split, candidate identity and evaluation settings | Fresh guard tests |

## Historical public sample verification

The earlier session ran the original released arrays through fold 0 with mirroring disabled and raw outputs. The global plaque Dice values were 0.8162 for P01 CX, 0.7232 for P01 LAD, and 0.6075 for P01 RCA. Independence of these released samples from the authors' training data remains unknown. The values provide engineering verification from the earlier session. Public model inference was not rerun as part of this download reissue.

Released inputs use 44 × 44 cross-sections, identity affines, and stored unit spacing. Source deformation and physical equivalence to the generated reconstructions remain unresolved. The released model's label 1 denotes non-pathological vessel, while SHK label 1 denotes patent lumen.

## Material implementation choices

- Native screening defaults to conservative 64³ patches, batch 1 and accumulation 2. The allocator cap is 22 GiB. CUDA performance remains unmeasured.
- Automatically eligible patients retain pending visual anatomical review unless reviewed cases are explicitly required.
- Aorta-free root selection uses a configurable 1.20 local-caliber ratio. Its confidence remains heuristic.
- Small surfaces retain full density below 1,000 post-decimation points. Target-reach QC rejects truncated paths.
- Best checkpoints use deterministic validation patches. Full-volume evaluation supports model comparison.
- The Moosavi intensity offset and reconstruction equivalence require local validation.
- Healthy-branch specificity uses successful branches, with abstention and successful-negative yield recorded separately.
- Undefined surfaces retain null metrics and explicit counts.
- Learned gating uses held-out validation calibration. An out-of-fold generator remains incomplete.
- External trainer installation copies one Python source into the active nnU-Net package and rejects a conflicting existing file.

Private SHK validation and Windows/CUDA execution remain pending. Plaque-specific ImageCAS/ADE-HTL adaptation, general multitask training, dedicated boundary-patch mining, geodesic/vesselness and cross-sectional refinement, pairwise voxel error-overlap analysis, and complete automatic failure-image panels remain incomplete.

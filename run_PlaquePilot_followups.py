#!/usr/bin/env python
"""Validation-only probability tests, followed optionally by paired CTTA-off training.

No held-out test patients are loaded. No existing predictions/checkpoints are replaced.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from plaque_review_core import (atomic_json, discover_predictions, exclusive, read_json,
                               safe_name, source_paths)

MODELS = ['cas_net', 'ffr_unet', 'swin_unetr', 'nnunet', 'nnunet_resenc']


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp, path)


def file_identity(path):
    path = Path(path).resolve(); stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def identity_lock(path, value):
    if path.is_file() and read_json(path) != value:
        raise ValueError(f'Inputs/settings changed: use a new --out directory. Existing identity: {path}')
    if not path.exists(): atomic_json(path,value)


def candidate_specs(thresholds=(.35,.5,.65), volumes=(0.,1.)):
    sources = [('nnunet',), ('nnunet_resenc',), ('ffr_unet',), ('nnunet','nnunet_resenc'), ('nnunet','ffr_unet')]
    specs = [dict(name=f'{s[0]}_argmax', sources=list(s), threshold=None, min_volume_mm3=0.) for s in sources[:3]]
    for source in sources:
        for threshold in thresholds:
            for volume in volumes:
                specs.append(dict(name=f'{"+".join(source)}_t{threshold:g}_v{volume:g}', sources=list(source),
                                  threshold=threshold, min_volume_mm3=volume))
    return specs


def binary_candidate(score, coverage, spec, voxel_mm3, argmax=None):
    mask = (argmax.copy() if spec['threshold'] is None else coverage & (score >= spec['threshold']))
    if spec['min_volume_mm3'] > 0:
        labels, count = ndi.label(mask, ndi.generate_binary_structure(3,3))
        sizes = np.bincount(labels.ravel())*voxel_mm3
        keep = sizes >= spec['min_volume_mm3']; keep[0] = False
        mask = keep[labels]
    return mask


def paired_interval(candidate, reference, seed=2028):
    keys = sorted(set(candidate) & set(reference))
    differences = np.array([candidate[k]-reference[k] for k in keys],float)
    if not len(differences): return None,None,None,0
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0,len(differences),(5000,len(differences)))].mean(axis=1)
    low, high = np.quantile(means,[.025,.975])
    return float(differences.mean()), float(low), float(high), len(keys)


def select_sources(sets):
    result = {}
    for model in ('nnunet','nnunet_resenc','ffr_unet'):
        found = [p for p in sets if p.name.split('/')[-1] == model]
        if len(found) != 1:
            raise ValueError(f'Expected one {model} source, got {len(found)}. Use explicit --prediction {model}=DIRECTORY entries.')
        if found[0].task != 'multiclass':
            raise ValueError('This follow-up sweep requires the original multiclass model outputs.')
        result[model] = found[0]
    return result


def compact_case(manifest, case, sources, specs):
    """Read each source once per patient; crop scoring to the union of all possible positives."""
    from plaque_benchmark.data import branch_arrays, target_array
    from plaque_benchmark.prediction import read_label_prediction, read_prediction
    from plaque_benchmark.evaluation import branch_assignment
    row = manifest['cases'][case]; ref = nib.load(row['image'])
    truth = target_array(row,'multiclass') == 2
    branches = branch_arrays(row,ref)
    scores, coverage, argmax = {}, {}, {}
    union = truth.copy()
    minimum = min(spec['threshold'] for spec in specs if spec['threshold'] is not None)
    for model, prediction in sources.items():
        probs, covered = read_prediction(prediction.root,case,ref,'multiclass')
        scores[model] = probs[2].copy(); coverage[model] = covered
        argmax[model] = read_label_prediction(prediction.root,case,ref,'multiclass') == 2
        union |= (covered & (scores[model] >= minimum)) | argmax[model]
        del probs
    # Convex averages cannot exceed the largest member, so this union covers every candidate.
    assigned, coords = branch_assignment(union,branches,ref.affine)
    branch_grid = np.full(ref.shape,-1,np.int8)
    if len(coords): branch_grid[tuple(coords.T)] = assigned
    nonzero = np.nonzero(union)
    crop = tuple(slice(int(x.min()),int(x.max())+1) for x in nonzero) if union.any() else (slice(0,1),)*3
    healthy = [i for i,name in enumerate(('LAD','LCX','RCA')) if (branches[name] > 0).any() and not (branches[name] == 2).any()]
    voxel = abs(float(np.linalg.det(ref.affine[:3,:3])))
    return (truth[crop].copy(), {k:v[crop].copy() for k,v in scores.items()},
            {k:v[crop].copy() for k,v in coverage.items()}, {k:v[crop].copy() for k,v in argmax.items()},
            branch_grid[crop].copy(), healthy, voxel)


def score_case(manifest, case, sources, specs):
    from plaque_benchmark.evaluation import overlap, lesions
    truth,scores,coverage,argmax,assignment,healthy,voxel = compact_case(manifest,case,sources,specs)
    rows = []; combined = {}
    for spec in specs:
        source = tuple(spec['sources'])
        if source not in combined:
            combined[source] = (sum(scores[k] for k in source)/len(source),np.logical_and.reduce([coverage[k] for k in source]))
        score,support = combined[source]
        mask = binary_candidate(score,support,spec,voxel,argmax[source[0]])
        detected, details = lesions(mask,truth,voxel)
        healthy_fps = healthy_positive = 0
        for index in healthy:
            pred = mask & (assignment == index)
            healthy_positive += int(pred.any())
            healthy_fps += int(ndi.label(pred,ndi.generate_binary_structure(3,3))[1])
        small = [item for item in details if item['volume_mm3'] < 5]
        rows.append(dict(candidate=spec['name'],case=case,**overlap(mask,truth,voxel),**detected,
                         healthy_branches=len(healthy),healthy_positive=healthy_positive,healthy_fp_components=healthy_fps,
                         small_lesions=len(small),small_detected=sum(bool(item['detected']) for item in small)))
    return rows


def aggregate(rows, specs):
    def average(values):
        v = [x for x in values if x is not None]; return float(np.mean(v)) if v else None
    reference = {r['case']:r['dice'] for r in rows if r['candidate'] == 'nnunet_argmax' and r['dice'] is not None}
    summaries = []
    for spec in specs:
        selected = [r for r in rows if r['candidate'] == spec['name']]
        n = sum(r['healthy_branches'] for r in selected)
        small = sum(r['small_lesions'] for r in selected)
        lesions = sum(r['lesion_tp']+r['lesion_fn'] for r in selected)
        paired = {r['case']:r['dice'] for r in selected if r['dice'] is not None}
        delta,lo,hi,paired_n = paired_interval(paired,reference)
        summaries.append(dict(candidate=spec['name'],plaque_dice=average(r['dice'] for r in selected),
                             plaque_precision=average(r['precision'] for r in selected),plaque_recall=average(r['recall'] for r in selected),
                             positive_patients=len(paired),patients=len(selected),healthy_branch_fp_rate=sum(r['healthy_positive'] for r in selected)/n if n else None,
                             healthy_fp_components_per_branch=sum(r['healthy_fp_components'] for r in selected)/n if n else None,
                             lesion_sensitivity=sum(r['lesion_tp'] for r in selected)/lesions if lesions else None,
                             small_lesion_sensitivity=sum(r['small_detected'] for r in selected)/small if small else None,
                             small_lesions=small,healthy_branches=n,delta_dice_vs_nnunet=delta,paired_ci_low=lo,paired_ci_high=hi,paired_n=paired_n))
    return sorted(summaries,key=lambda r: -(r['plaque_dice'] if r['plaque_dice'] is not None else -1))


def export_candidate(manifest,sources,spec,out,export_probabilities=False):
    from plaque_benchmark.prediction import read_prediction, read_label_prediction, save_prediction
    root = Path(out)/'provisional_predictions'; root.mkdir(parents=True,exist_ok=True)
    identity_lock(root/'selection.json',spec)
    atomic_json(root/'run.json',dict(split_sha256=manifest['split_sha256'],partition='val',task='plaque',geometry='native',
                                  model='validation_followup',parameters=spec,label_only=not export_probabilities,
                                  selection_status='provisional: selected and assessed on the same validation patients'))
    for case,row in manifest['cases'].items():
        if row['partition'] != 'val': continue
        if (root/f'{case}.nii.gz').is_file() and (not export_probabilities or (root/f'{case}.npz').is_file()): continue
        print(f'Export provisional prediction: {case}',flush=True)
        ref = nib.load(row['image']); score = None; covered = None
        for model in spec['sources']:
            probs,support = read_prediction(sources[model].root,case,ref,'multiclass')
            score = probs[2].copy() if score is None else score+probs[2]
            covered = support if covered is None else covered & support
        score /= len(spec['sources'])
        argmax = read_label_prediction(sources[spec['sources'][0]].root,case,ref,'multiclass') == 2
        mask = binary_candidate(score,covered,spec,abs(np.linalg.det(ref.affine[:3,:3])),argmax)
        if export_probabilities:
            save_prediction(root,case,ref,np.stack([np.where(covered,1-score,0),np.where(covered,score,0)]),covered,seg=mask.astype(np.uint8))
        else:
            image = nib.Nifti1Image(mask.astype(np.uint8),ref.affine,ref.header.copy())
            image.set_data_dtype(np.uint8); nib.save(image,root/f'{case}.nii.gz')
    return root


def validation_experiments(args, manifest):
    sets = discover_predictions(args.bench,manifest,'val',args.runs,args.prediction)
    sources = select_sources(sets); specs = candidate_specs()
    cases = sorted(case for case,row in manifest['cases'].items() if row['partition']=='val')
    inputs = [args.dataset]
    for case in cases:
        row = manifest['cases'][case]; inputs += [Path(row['image']),Path(row['label']),*[Path(v) for v in row['branches'].values()]]
        for source in sources.values(): inputs += [source.root/'run.json',source.root/f'{case}.npz',source.root/f'{case}.nii.gz']
    identity = dict(schema=1,inputs=[file_identity(p) for p in inputs],specs=specs,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),export_probabilities=args.export_probabilities)
    identity_lock(args.out/'validation_identity.json',identity)
    cache = args.out/'case_cache'; cache.mkdir(exist_ok=True); rows = []
    for i,case in enumerate(cases):
        path = cache/(safe_name(case)+'.json')
        print(f'Validation patient {i+1}/{len(cases)}: {case} — {len(specs)} candidates'+(' [cached]' if path.exists() else ''),flush=True)
        if not path.exists(): atomic_json(path,score_case(manifest,case,sources,specs))
        rows += read_json(path)
    summaries = aggregate(rows,specs)
    write_csv(args.out/'validation_case_metrics.csv',rows); write_csv(args.out/'validation_candidates.csv',summaries)
    atomic_json(args.out/'validation_candidates.json',summaries)
    selected = summaries[0]; spec = next(s for s in specs if s['name'] == selected['candidate'])
    atomic_json(args.out/'provisional_winner.json',dict(spec=spec,metrics=selected,warning='Exploratory, same-validation selection. No test access or automatic promotion. Bootstrap intervals are unadjusted for selection/multiple comparisons.'))
    root = export_candidate(manifest,sources,spec,args.out,args.export_probabilities)
    print(f'\nProvisional highest validation Dice: {selected["candidate"]} ({selected["plaque_dice"]:.4f})\nReview FP rate and <5 mm3 lesion sensitivity before selecting it.\nMetrics: {args.out / "validation_candidates.csv"}\nViewer prediction directory: {root}',flush=True)


def paired_ablation(args):
    """Use each trained native model's actual settings; only CTTA is removed."""
    if args.prediction:
        raise ValueError('For training ablations use --runs with standard runner layouts; explicit predictions alone do not identify training configs.')
    manifest = read_json(args.dataset)
    sets = discover_predictions(args.bench,manifest,'val',args.runs)
    models = args.ablation_models
    if len(set(models)) != len(models): raise ValueError('Select each ablation model once')
    expected_configs = {}
    config_dir = args.out/'ctta_off_configs'; config_dir.mkdir(parents=True,exist_ok=True)
    for model in models:
        matches = [p for p in sets if p.name.split('/')[-1] == model]
        if len(matches)!=1: raise ValueError(f'Expected exactly one original {model} run for a paired ablation.')
        original = matches[0].root.parent.parent.parent/'training'/model
        if model.startswith('nnunet'):
            # The source runner records nnU-Net configuration in specification.json.
            context = read_json(original/'specification.json')
            cfg = context['context']['configuration']
            if cfg['epochs'] != args.epochs or cfg['steps'] != args.steps:
                raise ValueError(f'{model} original budget is {cfg["epochs"]} epochs / {cfg["steps"]} steps. Match --epochs/--steps for the paired ablation.')
        else:
            cfg = read_json(original/'override.json')
            if cfg['epochs'] != args.epochs or cfg['steps'] != args.steps:
                raise ValueError(f'{model} original budget differs; match --epochs / --steps.')
            cfg['augmentation'] = None
            identity_lock(config_dir/(model+'.json'),cfg)
        expected_configs[model] = {k:v for k,v in cfg.items() if k != 'augmentation'}
    launch = args.repo/'run_PlaquePilot.py'
    if not launch.is_file(): raise FileNotFoundError(f'Missing existing launcher: {launch}')
    source_files = sorted((args.repo/'plaque_benchmark').rglob('*.py')) + sorted((args.repo/'models').rglob('*.py')) + sorted((args.repo/'utils').rglob('*.py')) + [args.repo/'run_plaque_benchmark.sh', launch]
    identity_lock(args.out/'ablation_identity.json',dict(models=models,configs=expected_configs,common_run=args.ablation_prefix,
                    split=manifest['split_sha256'],sources={str(path.relative_to(args.repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}))
    failures = []
    logroot = args.out/'ablation_logs'; logroot.mkdir(exist_ok=True)
    common = ['--repo',str(args.repo),'--bench',str(args.bench),'--run-prefix',args.ablation_prefix,
              '--config-dir',str(config_dir),'--epochs',str(args.epochs),'--steps',str(args.steps)]
    for model in models:
        for mode in ('baseline-init','baseline-train'):
            command = [sys.executable,str(launch),mode,'--models',model,*common]
            marker = logroot/f'{model}_{mode}.done.json'
            if marker.is_file():
                receipt = read_json(marker)
                try: valid = receipt['command'] == command and receipt['artifacts'] == [file_identity(item[0]) for item in receipt['artifacts']]
                except FileNotFoundError: valid = False
                if valid:
                    print(f'Reused completed {model} {mode}',flush=True); continue
            print('\nCOMMAND '+subprocess.list2cmdline(command),flush=True)
            with (logroot/f'{model}_{mode}.log').open('a',encoding='utf-8') as log:
                env = dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1')
                process = subprocess.Popen(command,cwd=args.repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                try:
                    for line in process.stdout: print(line,end='',flush=True);log.write(line);log.flush()
                    status = process.wait()
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
                    raise
            if status:
                failures.append(dict(model=model,mode=mode,status=status)); break
            run_root = args.bench/'runner'/(args.ablation_prefix+'_baseline25')
            if mode == 'baseline-init':
                artifact = run_root/'init'/model/'PASS.json'
                init_receipt = read_json(artifact)
                actual = read_json(init_receipt['context'])['configuration']
                actual = {k:v for k,v in actual.items() if k != 'augmentation'}
                if actual != expected_configs[model]:
                    mismatch = logroot/f'{model}_configuration_mismatch.json'
                    atomic_json(mismatch,dict(expected=expected_configs[model],actual=actual))
                    print(f'Paired settings changed; training skipped. See {mismatch}',flush=True)
                    failures.append(dict(model=model,mode=mode,status='paired_configuration_mismatch')); break
            else:
                artifact = run_root/'reports'/model/'summary.json'
            artifacts = [artifact]
            if mode == 'baseline-init': artifacts.append(Path(init_receipt['context']))
            else:
                checkpoint = read_json(artifact).get('run',{}).get('checkpoint')
                if checkpoint: artifacts.append(Path(checkpoint))
            atomic_json(marker,dict(command=command,artifacts=[file_identity(item) for item in artifacts]))
    atomic_json(args.out/'ablation_status.json',dict(models=models,failures=failures,run=args.ablation_prefix+'_baseline25'))
    if failures: raise RuntimeError('Some ablations failed; see ablation_status.json and ablation_logs. Successful models are retained.')
    # Copy only small report files to give paired runs unambiguous plot labels.
    import shutil
    from plaque_benchmark.reporting import compare
    paired_reports = []
    for model in models:
        source = next(item for item in sets if item.name.split('/')[-1] == model)
        original_report = source.root.parent.parent.parent/'reports'/model
        off_report = args.bench/'runner'/(args.ablation_prefix+'_baseline25')/'reports'/model
        for condition,report in [('CTTA_on',original_report),('CTTA_off',off_report)]:
            copy = args.out/'paired_reports'/(model+'_'+condition); copy.mkdir(parents=True,exist_ok=True)
            for name in ('summary.json','metrics.json','lesions.json'):
                shutil.copy2(report/name,copy/name)
            summary = read_json(copy/'summary.json');summary['run']['display_name'] = model+' '+condition
            atomic_json(copy/'summary.json',summary);paired_reports.append(copy)
    compare(paired_reports,args.out/'ctta_paired_comparison')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',choices=['metrics','ablation','all'],default='metrics')
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--bench',type=Path,default=Path('experiments/PlaquePilot_confirmed'))
    p.add_argument('--dataset',type=Path)
    p.add_argument('--runs',nargs='+')
    p.add_argument('--prediction',action='append',help='NAME=DIRECTORY; use the names nnunet, nnunet_resenc, ffr_unet')
    p.add_argument('--out',type=Path,default=Path('experiments/PlaquePilot_followups_v1'))
    p.add_argument('--export-probabilities',action='store_true',help='Also export large NPZ files for the provisional winner; default exports viewer-ready labels only')
    p.add_argument('--ablation-models',nargs='+',choices=MODELS,default=['nnunet','nnunet_resenc','ffr_unet'])
    p.add_argument('--ablation-prefix',default='PlaquePilot_CTTAoff_paired80')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--steps',type=int,default=100)
    args = p.parse_args(argv); args.repo = args.repo.resolve()
    def resolve(path): return path.resolve() if path.is_absolute() else (args.repo/path).resolve()
    args.bench=resolve(args.bench);args.out=resolve(args.out)
    args.dataset=resolve(args.dataset) if args.dataset else args.bench/'prepared/dataset.json'
    if args.epochs < 1 or args.steps < 1: p.error('Epochs and steps must be positive')
    sys.path.insert(0,str(args.repo)); os.chdir(args.repo)
    args.out.mkdir(parents=True,exist_ok=True)
    from plaque_benchmark.data import read_json as read_manifest
    manifest = read_manifest(args.dataset)
    sleep_state = False
    try:
        if os.name=='nt':
            import ctypes
            sleep_state = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        with exclusive(args.out/'.followups.lock'):
            if args.mode in ('metrics','all'): validation_experiments(args,manifest)
            if args.mode in ('ablation','all'): paired_ablation(args)
    finally:
        if sleep_state: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: sys.exit(130)

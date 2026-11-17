"""PlaquePilot: confirmed-cohort preparation and staged experiments on Windows.

Default invocation performs preparation and initialization only. Training and
Optuna require their own explicit command. Place beside pyproject.toml.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from datetime import datetime, timezone

INSPECTED_COMMIT='c597d3334d957f987502a1a0f612738ae7e4a1fc'
MODELS=('cas_net','ffr_unet','swin_unetr','nnunet','nnunet_resenc')
NATIVE=('cas_net','ffr_unet','swin_unetr')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    temporary.replace(path)


def file_record(path):
    path=Path(path).resolve();st=path.stat()
    return dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns)


def physical_mask(image):
    import numpy as np
    import nibabel as nib
    if image.ndim!=3:
        raise ValueError('Masks must be scalar 3D NIfTIs')
    units=image.header.get_xyzt_units()[0]
    scale={'mm':1.,'unknown':1.,'meter':1000.,'micron':.001}.get(units)
    if scale is None or not (int(image.header['qform_code']) or int(image.header['sform_code'])):
        raise ValueError('Mask requires a coded physical transform with supported spatial units')
    affine=image.affine.copy();affine[:3,:]*=scale
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-12:
        raise ValueError('Mask requires a finite, invertible physical transform')
    return nib.Nifti1Image(np.asanyarray(image.dataobj),affine)


def cohort_sources(images,masks):
    from plaque_benchmark.data import discover,nifti_paths
    found=discover(images,masks)
    assigned=set(p for source in found.values() for p in source['images'])
    unknown=[str(p.resolve()) for p in nifti_paths(images) if str(p.resolve()) not in assigned]
    if unknown:
        raise ValueError('CT files lack unambiguous SHK case IDs: '+str(unknown))
    errors=[]
    for case,source in found.items():
        if len(source['images'])!=1:
            errors.append(f'{case}: expected one CT, found {len(source["images"])}')
        for branch in ('LAD','LCX','RCA'):
            if len(source['masks'].get(branch,[]))!=1:
                errors.append(f'{case}/{branch}: expected one mask')
    if errors:
        raise ValueError('No cases have been dropped. Resolve input pairing:\n'+'\n'.join(errors))
    return found


def prepare_confirmed(images,masks,bench,*,seed=2028,reuse_split=None,fresh_split=False):
    """Authorize all discovered cases based on the user's visual confirmation."""
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to
    from tqdm.auto import tqdm
    from plaque_benchmark.data import (BRANCHES,digest,file_sha256,geometry,label_array,make_split,save_image)
    bench=Path(bench).resolve();sources=cohort_sources(images,masks)
    if len(sources)<3:raise ValueError('At least three patients are required')
    input_files=[file_record(p) for source in sources.values()
                 for p in source['images']+sum((source['masks'][b] for b in BRANCHES),[])]
    request=dict(images=str(Path(images).resolve()),masks=str(Path(masks).resolve()),
                 files=input_files,seed=seed,policy='user_confirmed_physical_alignment',
                 reuse_split=str(Path(reuse_split).resolve()) if reuse_split else None,
                 reuse_split_sha256=file_sha256(reuse_split) if reuse_split else None,
                 fresh_split=bool(fresh_split))
    manifest_path=bench/'prepared/dataset.json';receipt_path=bench/'approval_receipt.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        if receipt['request']!=request:
            raise ValueError('Dataset inputs or split policy changed. Use a new --bench; the locked cohort was not overwritten.')
        if not manifest_path.is_file():raise ValueError('Prepared manifest missing; use a new --bench or restore the existing cohort')
        if file_sha256(manifest_path)!=receipt['manifest_sha256']:
            raise ValueError('Prepared manifest changed after approval')
        for asset in receipt['derived_assets']:
            if file_record(asset['path'])!=asset:
                raise ValueError('Prepared label asset changed or is missing: '+asset['path'])
        print(f'Reused confirmed cohort: {len(sources)} patients',flush=True)
        return read(manifest_path)
    progress_file=bench/'approval_request.json'
    if progress_file.exists() and read(progress_file)!=request:
        raise ValueError('An unfinished preparation belongs to different inputs; choose a new --bench')
    write(progress_file,request)
    report=dict(schema=1,cases=[],manual_accept_overrides=True,
                approval_basis='User confirmed MasksUnverified matches AsNifti; all discovered cases accepted',
                geometry_policy='Use original physical coordinates; nearest-neighbor resampling only when grids differ',
                qc_scope='Input pairing, finite CT, numeric labels, physical geometry, empty targets and overlaps; visual alignment confirmed by user; topology not re-audited')
    errors=[]
    for case,source in tqdm(sources.items(),desc='Approve/build cases',unit='case',ascii=True):
        checkpoint=bench/'case_receipts'/f'{case}.json'
        if checkpoint.exists():
            cached=read(checkpoint)
            try:
                if all(file_record(a['path'])==a for a in cached['assets']):
                    report['cases'].append(cached['case']);continue
            except OSError:pass
        try:
            image_path=source['images'][0];ref=nib.load(image_path);geometry(ref)
            ct=ref.get_fdata(dtype=np.float32,caching='unchanged')
            if not np.isfinite(ct).all():raise ValueError('CT contains nonfinite values')
            del ct
            row=dict(case=case,image=image_path,review='accept',status='eligible',accept_override=True,
                     image_sha256=file_sha256(image_path),errors=[],flags=[],branches={})
            combined=np.zeros(ref.shape,np.uint8);branch_lumen=np.zeros(ref.shape,np.uint8)
            arrays=[]
            for index,branch in enumerate(BRANCHES,1):
                original=source['masks'][branch][0];mask=nib.load(original);label_array(mask)
                physical=physical_mask(mask)
                if int(mask.header['qform_code']) and int(mask.header['sform_code']) and not np.allclose(mask.get_qform(),mask.get_sform(),atol=.005,rtol=0):
                    row['flags'].append(f'{branch}: qform/sform conflict retained; NIfTI-selected affine used')
                same=mask.shape==ref.shape and np.allclose(physical.affine,ref.affine,atol=.005,rtol=0)
                labels=label_array(physical) if same else label_array(resample_from_to(physical,(ref.shape,ref.affine),order=0,mode='constant',cval=0))
                if not same:row['flags'].append(f'{branch}: physically resampled onto CT grid')
                if (label_array(mask)>0).any() and not (labels>0).any():
                    row['flags'].append(f'{branch}: nonempty source became empty on CT grid; retained by acceptance policy')
                if not (labels==1).any():row['flags'].append(f'{branch}: empty or no-lumen annotation retained')
                path=bench/'accepted_branches'/case/(branch+'.nii.gz');save_image(path,labels,ref.affine)
                row['branches'][branch]=dict(path=str(path),source_path=original,source_sha256=file_sha256(original),
                    lumen_voxels=int((labels==1).sum()),plaque_voxels=int((labels==2).sum()),resampled=not same)
                for previous in arrays:
                    overlap=(previous>0)&(labels>0)
                    if overlap.any():row['flags'].append(f'{branch}: overlapping branch voxels retained; plaque wins class conflicts')
                arrays.append(labels)
                np.maximum(combined,labels,out=combined)
                lumen=labels==1;branch_lumen[lumen]=np.where(branch_lumen[lumen]==0,index,255)
            label=bench/'prepared/labels'/f'{case}.nii.gz';branch_path=bench/'prepared/branch_lumen'/f'{case}.nii.gz'
            save_image(label,combined,ref.affine);save_image(branch_path,branch_lumen,ref.affine)
            row.update(label=str(label),branch_lumen=str(branch_path))
            assets=[file_record(p) for p in [label,branch_path]+[row['branches'][b]['path'] for b in BRANCHES]]
            write(checkpoint,dict(case=row,assets=assets));report['cases'].append(row)
        except (ValueError,OSError,KeyError,nib.filebasedimages.ImageFileError) as exc:
            errors.append(f'{case}: {exc}')
    if errors:
        write(bench/'input_errors.json',errors)
        raise ValueError('No partial cohort will be trained. Input requirements:\n'+'\n'.join(errors))
    if any(file_record(item['path'])!=item for item in input_files):
        raise ValueError('Source files changed during preparation; no approved manifest was created. Choose a new --bench after inputs are stable.')
    assert {r['case'] for r in report['cases']}==set(sources)
    split=make_split(report,seed=seed,require_reviewed=True)
    if reuse_split and not fresh_split:
        old=read(reuse_split);parts=old['partitions']
        flattened=[c for name in ('train','val','test') for c in parts[name]]
        if len(flattened)!=len(set(flattened)) or set(flattened)!=set(sources) or any(not parts[n] for n in ('train','val','test')):
            raise ValueError('Previous split covers a different cohort. No case has been excluded. Use a new --bench with --fresh-split for a separate experiment.')
        split['partitions']=parts;split['reused_from']=str(Path(reuse_split).resolve())
    write(bench/'audit.json',report);write(bench/'split.json',split)
    with (bench/'audit.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['case','review','status','flags']);writer.writeheader()
        for row in report['cases']:writer.writerow({k:('; '.join(row[k]) if k=='flags' else row[k]) for k in writer.fieldnames})
    manifest=dict(schema=1,split_sha256=digest(split),cases={})
    parts={case:name for name,ids in split['partitions'].items() for case in ids}
    assets=[]
    for row in report['cases']:
        case=row['case'];manifest['cases'][case]=dict(partition=parts[case],image=row['image'],image_sha256=row['image_sha256'],
            label=row['label'],branch_lumen=row['branch_lumen'],branches={b:row['branches'][b]['path'] for b in BRANCHES},
            annotation_status='accept',accept_override=True,original_audit_flags=row['flags'],geometry_policy='physical_coordinates')
        assets.extend(read(bench/'case_receipts'/f'{case}.json')['assets'])
    write(manifest_path,manifest)
    write(receipt_path,dict(request=request,manifest_sha256=file_sha256(manifest_path),derived_assets=assets,
        accepted_cases=sorted(sources),partitions={k:len(v) for k,v in split['partitions'].items()},
        approved_utc=datetime.now(timezone.utc).isoformat(),approval_basis='Explicit user confirmation of alignment in conversation',
        numerical_labels=[0,1,2],overlap_policy='plaque_priority',no_original_files_modified=True))
    print(f'Accepted all {len(sources)} patients. '+str({k:len(v) for k,v in split['partitions'].items()}),flush=True)
    return manifest


def load_runner(repo):
    path=Path(repo)/'run_plaque_benchmark.sh'
    content=path.read_text(encoding='utf-8')
    opening="<<'PYTHON_RUNNER'\n"
    if opening not in content or '\nPYTHON_RUNNER' not in content:
        raise ValueError('Unsupported run_plaque_benchmark.sh format; restore the inspected PlaquePilot launcher')
    body=content.split(opening,1)[1].rsplit('\nPYTHON_RUNNER',1)[0]
    module=types.ModuleType('plaque_pilot_embedded_runner');module.__file__=str(path)
    exec(compile(body,str(path),'exec'),module.__dict__)
    return module,path


def preflight(repo,*,external=True,device='cuda',allow_other_env=False):
    import importlib
    from importlib.metadata import version
    from plaque_benchmark.data import file_sha256
    if not allow_other_env and Path(sys.prefix).name.lower()!='plaquepilot':
        raise ValueError('Activate PlaquePilot before running this script. Python is '+sys.executable)
    required=['pyproject.toml','run_plaque_benchmark.sh','requirements_workstation.txt','plaque_benchmark/native.py',
              'plaque_benchmark/ctta.py','plaque_benchmark/optuna_search.py','configs/ctta_mild.json',
              'plaque_benchmark/configs/screen.json']
    for name in required:
        if not (Path(repo)/name).is_file():raise FileNotFoundError('Missing required repository file: '+name)
    result=dict(python=sys.executable,prefix=sys.prefix,inspected_commit=INSPECTED_COMMIT)
    for name,distribution in [('torch','torch'),('nibabel','nibabel'),('SimpleITK','SimpleITK'),('monai','monai'),('ctaug','ctaug'),('optuna','optuna')]:
        try:mod=importlib.import_module(name)
        except ImportError as exc:
            remedy='python -m pip install monai==1.5.2' if name=='monai' else 'python -m pip install -e ".[plaque]" and python -m pip install -r requirements_workstation.txt'
            raise ImportError(f'{name} is not importable in {sys.executable}. Remedy: {remedy}') from exc
        result[name]=dict(version=version(distribution),path=str(Path(mod.__file__).resolve()))
    import torch
    result['cuda_available']=torch.cuda.is_available()
    result['torch_cuda_build']=torch.version.cuda
    from plaque_benchmark.workstation import host_resources
    result['host']=host_resources()
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('This interpreter cannot use CUDA; fix the PlaquePilot PyTorch installation before launching GPU jobs')
    if torch.cuda.is_available():result.update(gpu=torch.cuda.get_device_name(0),gpu_vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    native=importlib.import_module('plaque_benchmark.native')
    result['native_source']=str(Path(native.__file__).resolve())
    if not Path(native.__file__).resolve().is_relative_to(Path(repo).resolve()):
        raise RuntimeError('Editable package points to another checkout; run python -m pip install -e ".[plaque]" here')
    if external:
        try:
            import nnunetv2
            from importlib.metadata import distribution
            package=Path(next(iter(nnunetv2.__path__))).resolve()
            entries={ep.name for ep in distribution('nnunetv2').entry_points if ep.group=='console_scripts'}
            if not {'nnUNetv2_train','nnUNetv2_predict','nnUNetv2_plan_and_preprocess'}<=entries:
                raise ValueError('nnU-Net console entry points are incomplete')
            local=Path(repo)/'nnunet'
            if local.is_dir() and not package.is_relative_to(local.resolve()):
                raise ValueError('nnU-Net imports from '+str(package)+' instead of your local nnunet folder. Run python -m pip install -e .\\nnunet')
            directory=package/'training/nnUNetTrainer/variants/training_length'
            if not directory.is_dir() or not os.access(directory,os.W_OK):
                raise ValueError('nnU-Net trainer installation directory must be writable: '+str(directory))
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
            import inspect
            if 'patch_size' not in inspect.signature(nnUNetTrainer.get_training_transforms).parameters:
                raise ValueError('Unsupported nnU-Net training-transform API; the CTTA adapter was verified against nnU-Net 2.8')
            result['nnunetv2']=dict(version=version('nnunetv2'),path=str(package))
        except ImportError as exc:raise ImportError('Install your local nnunet checkout into PlaquePilot with python -m pip install -e .\\nnunet') from exc
    try:
        result['git_branch']=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
        result['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    except (OSError,subprocess.CalledProcessError):pass
    print(json.dumps(result,indent=2),flush=True)
    return result


def nn_settings(bench,repo,manifest,explicit_id=None):
    from plaque_benchmark.data import digest
    state_path=Path(bench)/'nnunet_execution.json'
    if state_path.exists():
        state=read(state_path)
        if state['dataset_sha256']!=digest(manifest) or explicit_id is not None and explicit_id!=state['dataset_id']:
            raise ValueError('Locked nnU-Net execution settings differ; use a new --bench')
    else:
        paths={key:str(Path(os.environ.get(key) or (Path(repo)/key)).resolve())
               for key in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results')}
        for path in paths.values():Path(path).mkdir(parents=True,exist_ok=True)
        def available(index):
            return not any(list(Path(path).glob(f'Dataset{index:03d}_*')) for path in paths.values())
        if explicit_id is not None:
            if not 1<=explicit_id<=999:raise ValueError('nnU-Net dataset ID must be 1-999')
            if not available(explicit_id):raise ValueError('Requested dataset ID is occupied; omit --nnunet-id to allocate a free ID')
            index=explicit_id
        else:
            index=next((i for i in range(710,800) if available(i)),None)
            if index is None:raise ValueError('No free dataset ID in 710-799; choose --nnunet-id explicitly')
        state=dict(dataset_id=index,paths=paths,dataset_sha256=digest(manifest))
        write(state_path,state)
    # Reapply persisted paths rather than relying on shell variables on reruns.
    os.environ.update(state['paths'])
    os.environ['PLAQUE_PYTHON']=sys.executable;os.environ['NNUNET_PYTHON']=sys.executable
    return state


def runner_for(module,launcher,args,bench,state,mode,models=None,baseline=False,tuned=False):
    options=[mode,'--repo',str(args.repo),'--bench',str(bench),'--run-name',
             args.run_prefix+('_baseline25' if baseline else '_tuned_ctta25' if tuned else '_ctta25'),
             '--a5000','--workstation','--device',args.device,'--models',*(models or args.models),
             '--nnunet-python',sys.executable,'--preprocess-workers',str(args.preprocess_workers)]
    if state:options+=['--nnunet-id',str(state['dataset_id'])]
    if args.epochs:options+=['--epochs',str(args.epochs)]
    if args.steps:options+=['--steps',str(args.steps)]
    options+=['--init-steps',str(args.init_steps)]
    if tuned:options+=['--config-dir',str(args.search_out/'best_config')]
    elif args.config_dir:options+=['--config-dir',str(args.config_dir)]
    if not baseline and not tuned:options+=['--ctta-config',str(args.ctta_config)]
    if mode=='train':options+=['--resume']
    return module.Runner(module.parse(options),launcher)


def ensure_preprocessed(runner,models):
    external=[m for m in models if m in ('nnunet','nnunet_resenc')]
    if not external:return
    try:
        for model in external:runner.nn_configuration(model)
    except (FileNotFoundError,KeyError):
        print('Preparing nnU-Net export and both required plan configurations...',flush=True)
        runner.prepare_nnunet()
        for model in external:runner.nn_configuration(model)
    else:print('Reused matching nnU-Net export, plans, data and locked split.',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',default='init',choices=['doctor','prepare','init','train','baseline-init','baseline-train','search','tuned-init','tuned-train'])
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--images',type=Path,default=Path(r'D:\Data\PlaqueDataset\AsNifti'))
    parser.add_argument('--masks',type=Path,default=Path(r'D:\Data\PlaqueDataset\MasksUnverified'))
    parser.add_argument('--bench',type=Path,default=Path('experiments/PlaquePilot_confirmed'))
    parser.add_argument('--reuse-split',type=Path,help='Default: reuse old physical-cohort split if present')
    parser.add_argument('--fresh-split',action='store_true',help='Explicit new patient split; use a new --bench')
    parser.add_argument('--seed',type=int,default=2028)
    parser.add_argument('--run-prefix',default='PlaquePilot')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--ctta-config',type=Path,default=Path('configs/ctta_mild.json'))
    parser.add_argument('--config-dir',type=Path)
    parser.add_argument('--nnunet-id',type=int)
    parser.add_argument('--preprocess-workers',type=int,default=4)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--init-steps',type=int,default=100)
    parser.add_argument('--search-out',type=Path,default=Path('experiments/PlaquePilot_optuna_v1'))
    parser.add_argument('--trials',type=int,default=12)
    parser.add_argument('--search-epochs',type=int,default=8)
    parser.add_argument('--search-hours',type=float,default=4.)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--allow-other-env',action='store_true',help='Diagnostic/CPU-test override; normal use requires PlaquePilot')
    args=parser.parse_args(argv)
    args.repo=args.repo.resolve()
    def resolve(path):return path.resolve() if path.is_absolute() else (args.repo/path).resolve()
    args.bench=resolve(args.bench);args.ctta_config=resolve(args.ctta_config);args.search_out=resolve(args.search_out)
    if args.config_dir:args.config_dir=resolve(args.config_dir)
    if args.reuse_split:args.reuse_split=resolve(args.reuse_split)
    if args.fresh_split and args.reuse_split:parser.error('Choose --fresh-split or --reuse-split')
    if len(args.models)!=len(set(args.models)):parser.error('Select each model once')
    for key in ('seed','preprocess_workers','init_steps','trials','search_epochs'):
        if getattr(args,key)<(0 if key=='seed' else 1):parser.error(f'Invalid --{key.replace("_","-")}')
    for key in ('epochs','steps'):
        if getattr(args,key) is not None and getattr(args,key)<1:parser.error(f'Invalid --{key}')
    if args.search_hours<=0:parser.error('Search hours must be positive')
    sys.path.insert(0,str(args.repo));os.chdir(args.repo)
    args.bench.mkdir(parents=True,exist_ok=True)
    # Serialize coordinators: two foreground jobs should not compete for this GPU/cohort.
    from plaque_benchmark.optuna_search import study_lock
    try:
        with study_lock(args.repo/'experiments/PlaquePilot_execution.lock'):
            external=args.mode not in ('prepare','search','tuned-init','tuned-train') and any(m.startswith('nnunet') for m in args.models)
            env=preflight(args.repo,external=external,device=args.device,allow_other_env=args.allow_other_env)
            write(args.bench/'local_installation.json',env)
            if args.mode=='doctor':return 0
            if not args.reuse_split and not args.fresh_split:
                previous=args.repo/'experiments/plaque_all_accepted_physical/split.json'
                if previous.exists():args.reuse_split=previous
            manifest=prepare_confirmed(args.images,args.masks,args.bench,seed=args.seed,
                                       reuse_split=args.reuse_split,fresh_split=args.fresh_split)
            if args.mode=='prepare':return 0
            module,launcher=load_runner(args.repo)
            if args.mode=='search':
                models=[m for m in args.models if m in NATIVE]
                if not models:raise ValueError('Optuna currently supports native models; select cas_net, ffr_unet or swin_unetr')
                print('Native-only Optuna search: '+', '.join(models),flush=True)
                command=[sys.executable,'-m','plaque_benchmark.optuna_search','--dataset',str(args.bench/'prepared/dataset.json'),
                         '--out',str(args.search_out),'--models',*models,'--config-dir',str(args.config_dir or args.repo/'configs/plaque_workstation'),
                         '--ctta-config',str(args.ctta_config),'--trials',str(args.trials),'--epochs',str(args.search_epochs),
                         '--steps',str(args.steps or 100),'--timeout-hours',str(args.search_hours),'--device',args.device]
                # Passing this child our source path avoids a stale editable install in spawned processes.
                environment=dict(os.environ);environment['PYTHONPATH']=str(args.repo)+os.pathsep+environment.get('PYTHONPATH','')
                subprocess.run(command,cwd=args.repo,env=environment,check=True)
                return 0
            tuned=args.mode.startswith('tuned-');baseline=args.mode.startswith('baseline-')
            models=args.models
            if tuned:
                models=[m for m in args.models if m in NATIVE and (args.search_out/'best_config'/(m+'.json')).exists()]
                if not models:raise FileNotFoundError('No exported Optuna configurations found; run search first')
                print('Selected available tuned native models: '+', '.join(models),flush=True)
            state=nn_settings(args.bench,args.repo,manifest,args.nnunet_id) if any(m.startswith('nnunet') for m in models) else None
            mode='train' if args.mode.endswith('train') else 'init'
            runner=runner_for(module,launcher,args,args.bench,state,mode,models,baseline,tuned)
            if mode=='init':ensure_preprocessed(runner,models)
            elif state:
                # Training never triggers a hidden re-export/re-plan or bypasses init.
                for model in models:
                    if model.startswith('nnunet'):runner.nn_configuration(model)
            runner.execute()
            print('Initialization complete; review its summary before the separate train command.' if mode=='init' else 'Sequential experiment completed.',flush=True)
    except Exception as exc:
        write(args.bench/'execution_failure.json',dict(mode=args.mode,error=str(exc),utc=datetime.now(timezone.utc).isoformat()))
        raise
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Completed checkpoints and prepared cases remain reusable.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print('PlaquePilot stopped: '+str(exc),file=sys.stderr)
        raise SystemExit(1)

"""Unattended five-model PlaquePilot run; needs the execution-launcher overlay."""
from contextlib import contextmanager,redirect_stdout,redirect_stderr
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,os,subprocess,sys,time,traceback
import run_PlaquePilot as pilot

MODELS=pilot.MODELS


def utc():return datetime.now(timezone.utc).isoformat()


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()
    def isatty(self):return False


@contextmanager
def keep_awake():
    kernel=None
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        if not kernel.SetThreadExecutionState(0x80000001):
            raise OSError('Could not prevent Windows idle sleep; set plugged-in sleep to Never before running')
    try:yield
    finally:
        if kernel:kernel.SetThreadExecutionState(0x80000000)


class Workflow:
    def __init__(self,folder,identity):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'workflow.json'
        self.state=pilot.read(self.path) if self.path.exists() else dict(identity=identity,stages={},created_utc=utc())
        if self.state['identity']!=identity:
            raise ValueError('Unattended settings/code/data changed. Choose a new --run-prefix and --search-out; previous results were preserved.')
        self.save()
    def save(self):
        self.state['updated_utc']=utc();pilot.write(self.path,self.state)
    def stage(self,name,action):
        from plaque_benchmark.data import file_sha256
        old=self.state['stages'].get(name,{})
        if old.get('status') in ('PASS','CORRUPTED'):
            try:
                valid=all(Path(p).is_file() and file_sha256(p)==sha for p,sha in old.get('artifacts',{}).items())
                valid=valid and all(pilot.file_record(item['path'])==item for item in old.get('large_assets',[]))
            except OSError:valid=False
            if valid:
                old['status']='PASS';self.state['stages'][name]=old;self.save()
                print('REUSED '+name,flush=True);return old
            # Do not silently overwrite changed completed checkpoints or winners.
            result={**old,'status':'CORRUPTED','error':'A completed-stage artifact changed or is missing; restore it or use a new run prefix','updated_utc':utc()}
            self.state['stages'][name]=result;self.save()
            raise ValueError(name+': '+result['error'])
        self.state['stages'][name]=dict(status='RUNNING',started_utc=utc());self.save()
        print('\nSTART '+name,flush=True);start=time.perf_counter()
        try:
            result=action() or {}
            paths=result.pop('artifact_paths',[])
            result['artifacts']={str(Path(p).resolve()):file_sha256(p) for p in paths}
            result.update(status='PASS',seconds=time.perf_counter()-start,completed_utc=utc())
        except Exception as exc:
            traceback.print_exc()
            result=dict(status='FAIL',error=str(exc),seconds=time.perf_counter()-start,completed_utc=utc())
        self.state['stages'][name]=result;self.save()
        print(result['status']+' '+name,flush=True)
        return result


def execute_runner(args,module,launcher,state,mode,model,tuned=False):
    runner=pilot.runner_for(module,launcher,args,args.bench,state,mode,[model],tuned=tuned)
    if model.startswith('nnunet'):runner.nn_configuration(model)
    runner.execute()
    row=runner.records[0]
    paths=[runner.receipt(model)]
    if row.get('checkpoint'):paths.append(row['checkpoint'])
    if row.get('report'):paths.append(Path(row['report'])/'summary.json')
    return dict(record=row,report=row.get('report'),artifact_paths=paths)


def search_model(args,model):
    """Top up to the light trial budget; never add another full batch on restart."""
    import optuna
    out=args.search_out;db=out/'studies.sqlite3'
    tried=0
    if db.exists():
        try:
            study=optuna.load_study(study_name=model,storage='sqlite:///'+db.as_posix())
            tried=len(study.trials)
        except KeyError:pass
    remaining=max(0,args.trials-tried)
    if remaining:
        command=[sys.executable,'-m','plaque_benchmark.optuna_search','--dataset',str(args.bench/'prepared/dataset.json'),
            '--out',str(out),'--models',model,'--config-dir',str(args.config_dir or args.repo/'configs/plaque_workstation'),
            '--ctta-config',str(args.ctta_config),'--trials',str(remaining),'--epochs',str(args.search_epochs),
            '--steps',str(args.steps),'--timeout-hours',str(args.search_hours),'--device',args.device]
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1')
        env['PYTHONPATH']=str(args.repo)+os.pathsep+env.get('PYTHONPATH','')
        subprocess.run(command,cwd=args.repo,env=env,check=True)
    winner=out/'best_config'/(model+'.json');metadata=out/(model+'_best.json')
    if not winner.is_file() or not metadata.is_file():raise RuntimeError('Search did not export a valid winner')
    return dict(winner=str(winner),artifact_paths=[winner,metadata])


def run_models(args,flow,init,train,search):
    # Initialize independently so one model's software/OOM failure cannot skip all later models.
    ready={model:flow.stage('init/'+model,lambda m=model:init(m,False))['status']=='PASS' for model in args.models}
    outcomes={}
    # Run both external models before native tuning/long jobs, rather than leaving them untrained.
    for model in [m for m in args.models if m.startswith('nnunet')]:
        if not ready[model]:
            outcomes[model]=dict(status='FAIL',error='Initialization failed');continue
        outcomes[model]=flow.stage('train/'+model,lambda m=model:train(m,False))
    for model in [m for m in args.models if m in pilot.NATIVE]:
        if not ready[model]:
            outcomes[model]=dict(status='FAIL',error='Initialization failed');continue
        completed=next((name for name in ('tuned-train/'+model,'fallback-train/'+model)
                        if flow.state['stages'].get(name,{}).get('status') in ('PASS','CORRUPTED')),None)
        if completed:
            previous=flow.stage(completed,lambda:None)
            outcomes[model]={**previous,'configuration_source':'optuna_winner' if completed.startswith('tuned-') else 'fixed_ctta_fallback'}
            continue
        selection=flow.stage('search/'+model,lambda m=model:search(m))
        tuned=selection['status']=='PASS'
        if tuned:
            checked=flow.stage('tuned-init/'+model,lambda m=model:init(m,True))
            tuned=checked['status']=='PASS'
        # A failed search/winner check falls back visibly to the already verified fixed CTTA profile.
        # Never label fallback training as an Optuna winner.
        stage=('tuned-train/' if tuned else 'fallback-train/')+model
        result=flow.stage(stage,lambda m=model,t=tuned:train(m,t))
        result={**result,'configuration_source':'optuna_winner' if tuned else 'fixed_ctta_fallback'}
        outcomes[model]=result
    return outcomes


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--bench',type=Path,default=Path('experiments/PlaquePilot_confirmed'))
    p.add_argument('--images',type=Path,default=Path(r'D:\Data\PlaqueDataset\AsNifti'))
    p.add_argument('--masks',type=Path,default=Path(r'D:\Data\PlaqueDataset\MasksUnverified'))
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--trials',type=int,default=4);p.add_argument('--search-epochs',type=int,default=4)
    p.add_argument('--search-hours',type=float,default=1.)
    p.add_argument('--init-steps',type=int,default=100)
    p.add_argument('--run-prefix',default='PlaquePilot_away80')
    p.add_argument('--search-out',type=Path,default=Path('experiments/PlaquePilot_away_light_v1'))
    p.add_argument('--ctta-config',type=Path,default=Path('configs/ctta_mild.json'))
    p.add_argument('--config-dir',type=Path);p.add_argument('--reuse-split',type=Path)
    p.add_argument('--fresh-split',action='store_true');p.add_argument('--nnunet-id',type=int)
    p.add_argument('--preprocess-workers',type=int,default=4)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--allow-other-env',action='store_true',help='Diagnostic CPU tests only')
    args=p.parse_args(argv)
    if any(getattr(args,k)<1 for k in ('epochs','steps','trials','search_epochs','init_steps','preprocess_workers')) or args.search_hours<=0:
        p.error('Budgets and worker counts must be positive')
    if args.fresh_split and args.reuse_split:p.error('Choose --fresh-split or --reuse-split')
    if len(args.models)!=len(set(args.models)):p.error('Select each model once')
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_prefix):p.error('Invalid --run-prefix')
    args.repo=args.repo.resolve()
    for key in ('bench','search_out','ctta_config','config_dir','reuse_split'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,value.resolve() if value.is_absolute() else (args.repo/value).resolve())
    sys.path.insert(0,str(args.repo));os.chdir(args.repo)
    from plaque_benchmark.optuna_search import study_lock
    from plaque_benchmark.data import digest,file_sha256
    folder=args.bench/'unattended'/args.run_prefix;folder.mkdir(parents=True,exist_ok=True)
    with study_lock(args.repo/'experiments/PlaquePilot_execution.lock'),keep_awake(),(folder/'console.log').open('a',encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
            print('\nUNATTENDED RUN '+utc(),flush=True)
            env=pilot.preflight(args.repo,external=any(m.startswith('nnunet') for m in args.models),device=args.device,allow_other_env=args.allow_other_env)
            pilot.write(args.bench/'local_installation.json',env)
            if not args.reuse_split and not args.fresh_split:
                old=args.repo/'experiments/plaque_all_accepted_physical/split.json'
                if old.exists():args.reuse_split=old
            manifest=pilot.prepare_confirmed(args.images,args.masks,args.bench,seed=2028,reuse_split=args.reuse_split,fresh_split=args.fresh_split)
            state=pilot.nn_settings(args.bench,args.repo,manifest,args.nnunet_id) if any(m.startswith('nnunet') for m in args.models) else None
            module,launcher=pilot.load_runner(args.repo)
            source={str(path.relative_to(args.repo)):file_sha256(path) for sub in ('plaque_benchmark','models','utils')
                    for path in sorted((args.repo/sub).rglob('*.py'))}
            paths=[Path(__file__),args.repo/'run_PlaquePilot.py',launcher,args.ctta_config,
                   args.repo/'plaque_benchmark/configs/screen.json']
            paths+=sorted((args.config_dir or args.repo/'configs/plaque_workstation').glob('*.json'))
            # Free RAM/CPU availability changes each run; remove volatile host telemetry from identity.
            stable_env={k:v for k,v in env.items() if k not in ('host','git_commit','git_branch')}
            import ctaug
            stable_env['ctaug_source']={str(p.relative_to(Path(ctaug.__file__).parent)):file_sha256(p) for p in sorted(Path(ctaug.__file__).parent.rglob('*.py'))}
            if state:
                import nnunetv2
                package=Path(next(iter(nnunetv2.__path__)))
                stable_env['nnunet_source']={str(p.relative_to(package)):file_sha256(p) for p in sorted(package.rglob('*.py')) if not p.name.startswith('nnUNetTrainerSHKRunner')}
            identity=digest(dict(settings={k:str(v) for k,v in vars(args).items()},dataset=digest(manifest),source=source,
                                 controls={str(path):file_sha256(path) for path in paths},environment=stable_env,nn=state))
            flow=Workflow(folder,identity)
            external=[m for m in args.models if m.startswith('nnunet')]
            if external:
                def preprocessing():
                    runner=pilot.runner_for(module,launcher,args,args.bench,state,'init',external)
                    pilot.ensure_preprocessed(runner,external)
                    paths=runner.nn_paths();base=paths['nnUNet_preprocessed']/runner.dataset_name
                    assets=[];small=[]
                    for model in external:
                        info=runner.nn_configuration(model);small.append(base/(info['plan']+'.json'))
                        plan=pilot.read(small[-1]);data=base/plan['configurations']['3d_fullres']['data_identifier']
                        assets.extend(pilot.file_record(p) for p in data.iterdir() if p.suffix in ('.npz','.npy','.b2nd','.pkl'))
                    small += [base/'splits_final.json',paths['nnUNet_raw']/runner.dataset_name/'benchmark_export.json']
                    return dict(artifact_paths=small,large_assets=assets)
                flow.stage('preprocessing/nnunet',preprocessing)
            init=lambda model,tuned:execute_runner(args,module,launcher,state,'init',model,tuned)
            train=lambda model,tuned:execute_runner(args,module,launcher,state,'train',model,tuned)
            outcomes=run_models(args,flow,init,train,lambda model:search_model(args,model))
            pilot.write(folder/'model_summary.json',outcomes)
            with (folder/'model_summary.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['model','status','configuration_source','report','error'],extrasaction='ignore');writer.writeheader()
                for model,row in outcomes.items():writer.writerow(dict(model=model,**row))
            reports=[(model,r) for model,r in outcomes.items() if r['status']=='PASS' and r.get('report')]
            if reports:
                def comparison():
                    runner=pilot.runner_for(module,launcher,args,args.bench,state,'init',args.models)
                    import shutil
                    labelled=[]
                    for model,result in reports:
                        target=folder/'comparison_inputs'/model;target.mkdir(parents=True,exist_ok=True)
                        original=Path(result['report'])
                        for name in ('metrics.json','lesions.json'):
                            shutil.copyfile(original/name,target/name)
                        summary=pilot.read(original/'summary.json')
                        kind=result.get('configuration_source','planned_nnunet')
                        summary['run']['display_name']=model+' ('+kind+')'
                        pilot.write(target/'summary.json',summary);labelled.append(str(target))
                    runner.package('compare','--reports',*labelled,'--out',str(folder/'comparison'))
                    return dict(artifact_paths=[folder/'comparison/comparison.csv',folder/'comparison/comparison.json'])
                flow.stage('comparison',comparison)
            failed=[m for m in args.models if outcomes.get(m,{}).get('status')!='PASS']
            print('Finished. Failed/untrained models: '+str(failed)+'. Summary: '+str(folder/'model_summary.csv'),flush=True)
            if any(r.get('configuration_source')=='fixed_ctta_fallback' for r in outcomes.values()):
                print('Some native models used the fixed CTTA profile because search/winner initialization failed. See workflow.json.',flush=True)
            errors=[name for name,row in flow.state['stages'].items() if row.get('status') not in ('PASS',)]
            flow.state['model_coverage_complete']=not failed
            flow.state['failed_or_fallback_stages']=errors
            flow.save()
            return 1 if failed or any(name=='comparison' for name in errors) else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; rerun the same command to reuse completed stages and compatible checkpoints.',file=sys.stderr)
        raise SystemExit(130)

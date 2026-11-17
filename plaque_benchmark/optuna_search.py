"""Isolated, sequential native-model Optuna trials with epoch-level pruning.

Run python -m plaque_benchmark.optuna_search --help. No held-out test rows are
used by the objective. Best JSON files are configurations for fresh training.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import time

from .data import digest, read_json, write_json, file_sha256, training_signature


class TrialStopped(Exception):
    pass


@contextmanager
def study_lock(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    try:
        if stream.tell()==0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def score(row,metric):
    keys=['val_patch_dice_lumen','val_patch_dice_plaque'] if metric=='mean_dice' else ['val_patch_dice_plaque']
    values=[row.get(key) for key in keys]
    if any(v is None or not math.isfinite(float(v)) for v in values):
        raise ValueError('Search metric missing/nonfinite; confirm validation contains the target class')
    return sum(values)/len(values)


def trial_child(request_file):
    from .native import train
    request=read_json(request_file)
    folder=Path(request['folder'])
    def callback(row):
        epoch=row['epoch']
        print('OPTUNA_EPOCH='+json.dumps(dict(epoch=epoch,value=score(row,request['metric']))),flush=True)
        decision=folder/f'decision_{epoch}.json'
        deadline=time.monotonic()+60
        while not decision.exists():
            if (folder/'coordinator_interrupted.txt').exists():
                raise TrialStopped()
            if time.monotonic()>deadline:
                raise RuntimeError('Search coordinator did not acknowledge the epoch')
            time.sleep(.05)
        if read_json(decision)['prune']:
            raise TrialStopped()
    try:
        train(read_json(request['dataset']),'multiclass',request['config'],folder/'model',
              device=request['device'],epoch_callback=callback)
    except TrialStopped:
        return 42
    return 0


def run_trial(trial,request):
    folder=Path(request['folder']); folder.mkdir(parents=True)
    req=folder/'request.json'; write_json(req,request)
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1')
    process=subprocess.Popen([sys.executable,'-m','plaque_benchmark.optuna_search','_trial',str(req)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
    best=-float('inf')
    try:
        with (folder/'trial.log').open('w',encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith('OPTUNA_EPOCH='):
                    event=json.loads(line.split('=',1)[1]); best=max(best,event['value'])
                    trial.report(event['value'],event['epoch']-1)
                    prune=trial.should_prune()
                    decision=folder/f"decision_{event['epoch']}.json"
                    temporary=decision.with_suffix('.tmp')
                    write_json(temporary,dict(prune=prune))
                    temporary.replace(decision)
                    print(f"Trial {trial.number}, epoch {event['epoch']}: {event['value']:.4f}"+(' [prune]' if prune else ''),flush=True)
        status=process.wait()
    except BaseException:
        # Ask the child to exit at its next checkpoint where possible.
        (folder/'coordinator_interrupted.txt').write_text('Interrupted',encoding='utf-8')
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        raise
    if status==42:
        import optuna
        raise optuna.TrialPruned('Pruned after a completed checkpoint')
    if status!=0:
        raise RuntimeError(f'Trial failed with status {status}; inspect {folder / "trial.log"}')
    if not math.isfinite(best):
        raise RuntimeError('Trial produced no valid epoch metrics')
    return best


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='_trial':
        return trial_child(argv[1])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--models',nargs='+',choices=['cas_net','ffr_unet','swin_unetr','dynunet'],default=['cas_net'])
    p.add_argument('--config-dir',required=True,help='Native overrides, normally configs/plaque_workstation')
    p.add_argument('--ctta-config',help='Optional training augmentation policy')
    p.add_argument('--trials',type=int,default=12,help='Additional trials per model on each invocation')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--timeout-hours',type=float,default=4.,help='Budget per model; current trial finishes before timeout is checked')
    p.add_argument('--metric',choices=['plaque_dice','mean_dice'],default='plaque_dice')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args(argv)
    if min(args.trials,args.epochs,args.steps)<1 or args.timeout_hours<=0:
        p.error('Trials, epochs, steps and timeout must be positive')
    import optuna
    from .native import config_for, _validate_settings
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(args.dataset)
    # Full source fingerprints prevent silently mixing implementations in a study.
    repo=Path(__file__).resolve().parent.parent
    source={str(path.relative_to(repo)):file_sha256(path) for sub in ('plaque_benchmark','models','utils')
            for path in sorted((repo/sub).rglob('*.py'))}
    with study_lock(out.parent/'native_optuna_gpu.lock'):
        for model in args.models:
            override=Path(args.config_dir)/(model+'.json')
            base=config_for(model,'screen',str(override))
            if args.ctta_config:
                from .ctta import settings
                base['augmentation']=settings(read_json(args.ctta_config))
            _validate_settings(base,'multiclass')
            from importlib.metadata import version
            environment={'torch':version('torch'),'optuna':version('optuna')}
            if base.get('augmentation'):
                import ctaug
                environment['ctaug_sha256']=digest({str(p.relative_to(Path(ctaug.__file__).parent)):file_sha256(p) for p in sorted(Path(ctaug.__file__).parent.rglob('*.py'))})
            identity=digest(dict(schema=1,environment=environment,source=source,dataset=training_signature(manifest),base=base,
                                 metric=args.metric,epochs=args.epochs,steps=args.steps))
            db=out/'studies.sqlite3'
            study=optuna.create_study(study_name=model,storage='sqlite:///'+db.as_posix(),load_if_exists=True,
                direction='maximize',sampler=optuna.samplers.TPESampler(seed=base['seed']),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=4,n_warmup_steps=3,n_min_trials=3))
            old=study.user_attrs.get('identity')
            if old is not None and old!=identity:
                raise ValueError('Study code/data/budget/settings changed; select a new --out directory')
            study.sampler=optuna.samplers.TPESampler(seed=base['seed']+len(study.trials))
            study.set_user_attr('identity',identity)
            study.set_user_attr('metric_kind','validation_patch_screening')
            def objective(trial):
                cfg=deepcopy(base)
                cfg['lr']=trial.suggest_float('lr',1e-5,1e-3,log=True)
                cfg['weight_decay']=trial.suggest_float('weight_decay',1e-6,1e-3,log=True)
                cfg['sampling']['plaque']=trial.suggest_float('plaque_sampling',.20,.50)
                if cfg.get('augmentation'):
                    cfg['augmentation']['step']=trial.suggest_float('step_probability',0.,.20)
                    cfg['augmentation']['step_motion']=trial.suggest_float('step_motion_probability',0.,.20)
                # Export the original full-training epoch budget, not the short trial schedule.
                trial.set_user_attr('full_training_config',deepcopy(cfg))
                cfg.update(epochs=args.epochs,steps=args.steps,patience=0)
                _validate_settings(cfg,'multiclass')
                request=dict(dataset=str(Path(args.dataset).resolve()),config=cfg,
                             metric=args.metric,device=args.device,
                             folder=str(out/model/f'trial_{trial.number:05d}'))
                return run_trial(trial,request)
            study.optimize(objective,n_trials=args.trials,timeout=args.timeout_hours*3600,n_jobs=1,show_progress_bar=True)
            complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
            if complete:
                write_json(out/'best_config'/(model+'.json'),study.best_trial.user_attrs['full_training_config'])
                write_json(out/(model+'_best.json'),dict(trial=study.best_trial.number,score=study.best_value,
                           parameters=study.best_params,metric=args.metric,identity=identity,
                           requires_fresh_training_and_full_volume_validation=True))
            study.trials_dataframe().to_csv(out/(model+'_trials.csv'),index=False)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

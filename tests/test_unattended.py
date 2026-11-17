from pathlib import Path
import importlib.util,sys,types
import pytest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('away',HERE/'run_PlaquePilot_unattended.py')
away=importlib.util.module_from_spec(spec);spec.loader.exec_module(away)


def test_all_models_nn_training_and_failure_isolation(tmp_path):
    args=types.SimpleNamespace(models=list(away.MODELS))
    flow=away.Workflow(tmp_path/'flow','same');events=[]
    def init(model,tuned):events.append(('init',model,tuned));return {}
    def train(model,tuned):
        events.append(('train',model,tuned))
        if model=='nnunet':raise RuntimeError('one model fails')
        return {'report':str(tmp_path/model)}
    def search(model):
        events.append(('search',model))
        if model=='ffr_unet':raise RuntimeError('no winner')
        return {}
    result=away.run_models(args,flow,init,train,search)
    assert set(result)==set(away.MODELS)
    assert result['nnunet']['status']=='FAIL' and result['nnunet_resenc']['status']=='PASS'
    assert result['ffr_unet']['configuration_source']=='fixed_ctta_fallback'
    assert result['cas_net']['configuration_source']=='optuna_winner'
    assert next(i for i,e in enumerate(events) if e[:2]==('train','nnunet_resenc')) < next(i for i,e in enumerate(events) if e[0]=='search')
    events.clear()
    # Completed fixed-profile models stay completed; searches are not repeated behind them.
    result=away.run_models(args,flow,init,train,search)
    assert not any(e[:2]==('search','ffr_unet') for e in events)
    assert ('train','nnunet',False) in events  # Retry the failed model on resume.


def test_completed_artifacts_reused_and_tampering_stops(tmp_path):
    flow=away.Workflow(tmp_path/'state','fixed')
    asset=tmp_path/'winner.json';asset.write_text('{}')
    flow.stage('search/cas_net',lambda:{'artifact_paths':[asset]})
    assert flow.stage('search/cas_net',lambda:(_ for _ in ()).throw(AssertionError('should skip')))['status']=='PASS'
    asset.write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):flow.stage('search/cas_net',lambda:None)
    with pytest.raises(ValueError,match='settings/code/data changed'):away.Workflow(tmp_path/'state','different')


def test_real_cpu_workflow_search_winner_train_compare_and_resume(tmp_path,monkeypatch):
    import torch
    from plaque_benchmark.synthetic import create
    torch.set_num_threads(1)
    repo=Path(__import__('plaque_benchmark').__file__).resolve().parent.parent
    monkeypatch.chdir(repo)
    images,masks=create(tmp_path/'source',3)
    cfg=tmp_path/'configs';cfg.mkdir()
    away.pilot.write(cfg/'ffr_unet.json',dict(epochs=2,steps=1,patch=[16]*3,batch_size=1,accumulate=1,
         workers=0,cache_build_workers=2,model_params={'base_filters':4},validation_patches=2,patience=0))
    args=['--repo',str(repo),'--images',str(images),'--masks',str(masks),'--bench',str(tmp_path/'bench'),
          '--models','ffr_unet','--config-dir',str(cfg),'--epochs','2','--steps','1','--init-steps','1',
          '--trials','1','--search-epochs','1','--search-out',str(tmp_path/'search'),'--device','cpu','--allow-other-env']
    assert away.main(args)==0
    folder=tmp_path/'bench/unattended/PlaquePilot_away80'
    state=away.pilot.read(folder/'workflow.json')
    assert state['stages']['tuned-train/ffr_unet']['status']=='PASS'
    assert state['model_coverage_complete']
    assert (folder/'comparison/comparison.csv').is_file()
    import optuna
    url='sqlite:///'+(tmp_path/'search/studies.sqlite3').as_posix()
    n=len(optuna.load_study(study_name='ffr_unet',storage=url).trials)
    assert away.main(args)==0
    assert len(optuna.load_study(study_name='ffr_unet',storage=url).trials)==n

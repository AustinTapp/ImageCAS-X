import importlib.util
import json
from pathlib import Path
from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from plaque_benchmark.ctta import augment_hu,settings,NNUNetCTTA
from plaque_benchmark import native
from plaque_benchmark.data import audit,make_split,prepare,select,read_json,write_json
from plaque_benchmark.synthetic import create


def cfg(kind):
    result={k:0. for k in ('step','motion','step_motion','metal','wire','calcification')}
    result[kind]=1.
    if kind in ('metal','wire','calcification'):result['allow_synthetic_objects']=True
    return result


def sample():
    y=np.zeros((24,24,24),np.uint8);y[6:18,9:15,4:20]=1;y[10:13,12:15,9:12]=2
    x=np.where(y==1,350.,np.where(y==2,650.,-100.)).astype(np.float32)
    y[0]=255
    return x,y


@pytest.mark.parametrize('kind',['step','motion','step_motion','metal','wire','calcification'])
def test_actual_ctta_paths_and_repeatability(kind):
    x,y=sample(); before=x.copy();labels=y.copy()
    py=random.getstate();ns=np.random.get_state()
    a,b,info=augment_hu(x,y,(.4,.4,.6),cfg(kind),17)
    c,d,info2=augment_hu(x,y,(.4,.4,.6),cfg(kind),17)
    assert np.array_equal(a,c) and np.array_equal(b,d)
    assert a.shape==x.shape and b.dtype==y.dtype and np.isfinite(a).all()
    assert set(np.unique(b)) <= {0,1,2,255}
    assert np.array_equal(x,before) and np.array_equal(y,labels)
    assert random.getstate()==py
    assert np.array_equal(np.random.get_state()[1],ns[1])
    assert info and info2


def test_motion_image_label_and_ignore_move_together():
    x,y=sample();x=np.where(y==1,350.,np.where(y==2,650.,-100.)).astype(np.float32)
    a,b,info=augment_hu(x,y,(.4,.4,.6),cfg('motion'),37)
    assert np.all(a[b==1]==350) and np.all(a[b==2]==650)
    assert np.all(a[b==255]==-1024)


def test_object_insertion_requires_explicit_policy():
    with pytest.raises(ValueError,match='allow_synthetic_objects'):
        settings({'calcification':.1})


class Identity:
    def __call__(self,**data):return data


def test_nnunet_hu_roundtrip_and_unchanged_validation_contract():
    x,y=sample();seg=y.astype(np.int16);seg[seg==255]=-1
    mean,std=280.,170.
    transform=NNUNetCTTA(Identity(),{k:0. for k in ('step','motion','step_motion','metal','wire','calcification')},(.6,.4,.4),mean,std)
    out=transform(image=torch.from_numpy(((x-mean)/std)[None]),segmentation=torch.from_numpy(seg[None]))
    got=out['image'][0].numpy()*std+mean
    assert np.allclose(got,x,atol=1e-3)
    assert np.array_equal(out['segmentation'][0].numpy(),seg)


@pytest.fixture
def tiny(tmp_path):
    images,masks=create(tmp_path/'raw',3);report=audit(images,masks)
    return prepare(report,make_split(report),tmp_path/'prepared')


def small_config():
    cfg=native.config_for('ffr_unet','smoke')
    cfg.update(epochs=2,steps=1,patch=[16]*3,batch_size=1,accumulate=1,validation_patches=2,
               model_params={'base_filters':4},workers=0,cache_build_workers=2)
    return cfg


def test_parallel_cache_reuses_original_content_keys(tiny):
    rows=select(tiny,'train')+select(tiny,'val');c=small_config()
    a=native.prepare_patch_cache(rows,'multiclass',c);c['cache_build_workers']=1
    b=native.prepare_patch_cache(rows,'multiclass',c);assert a==b
    for case,path in a.items():assert (Path(path)/'complete.json').is_file()


def test_validation_never_augmented(tiny):
    rows=select(tiny,'val');c=small_config()
    kw=dict(training=False,cache_cases=1,sampling=c['sampling'])
    a=native.Patches(rows,'multiclass',[16]*3,4,2028,**kw)
    b=native.Patches(rows,'multiclass',[16]*3,4,2028,augmentation=cfg('step'),**kw)
    assert all(torch.equal(x,y) for x,y in zip(a[0],b[0]))


def test_augmented_train_callback_and_resume(tiny,tmp_path):
    torch.set_num_threads(1)
    c=small_config();c['augmentation']=cfg('step_motion');rows=[]
    out=tmp_path/'model'
    def callback(row):
        rows.append(row['epoch'])
        if row['epoch']==1:raise RuntimeError('interrupt after checkpoint')
    with pytest.raises(RuntimeError,match='interrupt after checkpoint'):
        native.train(tiny,'multiclass',c,out,device='cpu',epoch_callback=callback)
    native.train(tiny,'multiclass',c,out,device='cpu',resume=True,epoch_callback=lambda r:rows.append(r['epoch']))
    assert rows==[1,2]
    state=torch.load(out/'last.pt',weights_only=True)
    assert state['epoch']==2


def test_optuna_real_child_and_restarted_study(tiny,tmp_path):
    from plaque_benchmark.optuna_search import main
    config_dir=tmp_path/'cfg';config_dir.mkdir();write_json(config_dir/'ffr_unet.json',small_config())
    dataset=tmp_path/'dataset.json';write_json(dataset,tiny)
    args=['--dataset',str(dataset),'--out',str(tmp_path/'search'),'--models','ffr_unet',
          '--config-dir',str(config_dir),'--trials','1','--epochs','1','--steps','1','--device','cpu']
    assert main(args)==0 and main(args)==0
    import optuna
    study=optuna.load_study(study_name='ffr_unet',storage='sqlite:///'+(tmp_path/'search/studies.sqlite3').as_posix())
    assert len(study.trials)==2 and all(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    best=read_json(tmp_path/'search/best_config/ffr_unet.json');assert best['epochs']==2
    assert study.trials[0].params != study.trials[1].params


def test_pruning_stops_child_after_saved_epoch(tiny,tmp_path):
    import optuna
    from plaque_benchmark.optuna_search import run_trial
    class Prune:
        number=0
        def report(self,value,step):pass
        def should_prune(self):return True
    dataset=tmp_path/'dataset.json';write_json(dataset,tiny)
    request=dict(dataset=str(dataset),config=small_config(),metric='plaque_dice',device='cpu',folder=str(tmp_path/'trial'))
    with pytest.raises(optuna.TrialPruned):run_trial(Prune(),request)
    state=torch.load(tmp_path/'trial/model/last.pt',weights_only=True)
    assert state['epoch']==1

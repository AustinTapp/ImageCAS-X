import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plaque_review_core import (ReviewStore, checked_image, discover_predictions, load_volume, make_demo,
                               read_json, slice_array, source_paths, screen_to_indices)
from run_PlaquePilot_followups import binary_candidate, candidate_specs, aggregate, paired_interval


def save(path,data,affine=None):
    image = nib.Nifti1Image(data,np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units('mm'); nib.save(image,path); return path


@pytest.fixture
def tiny(tmp_path):
    ct = np.arange(5*6*7,dtype=np.float32).reshape(5,6,7)
    gt = np.zeros(ct.shape,np.uint8); gt[1,2,3] = 2; gt[2,3,4] = 1
    affine = np.diag([-.7,.8,1.2,1]);affine[0,3] = 8
    return dict(ct=save(tmp_path/'ct.nii.gz',ct,affine),
                ground_truth=save(tmp_path/'gt.nii.gz',gt,affine),
                prediction=save(tmp_path/'pred.nii.gz',gt,affine))


def test_canonical_geometry_and_physical_voxel_locations(tiny):
    volume = load_volume('case',tiny)
    assert nib.aff2axcodes(volume.shared.reference.affine) == ('R','A','S')
    assert volume.metrics['plaque']['dice'] == 1
    old = nib.load(tiny['ct']); new = volume.shared.reference
    world = old.affine@np.array([1,2,3,1]); ijk = np.rint(np.linalg.inv(new.affine)@world).astype(int)[:3]
    assert volume.shared.gt[tuple(ijk)] == 2
    assert volume.shared.ct[tuple(ijk)] == old.get_fdata()[1,2,3]


def test_mismatched_grid_is_explicit_and_resampled_physically(tiny):
    original = nib.load(tiny['prediction']); shifted = original.affine.copy();shifted[0,3] += .7
    save(tiny['prediction'],np.asanyarray(original.dataobj),shifted)
    with pytest.raises(ValueError,match='Prediction grid differs'): load_volume('case',tiny)
    volume = load_volume('case',tiny,resample_labels=True)
    assert any('Prediction grid differs' in w for w in volume.warnings)
    expected = original.affine@np.array([1,2,3,1]);expected[0] += .7
    voxel = np.rint(np.linalg.inv(volume.shared.reference.affine)@expected).astype(int)[:3]
    assert volume.pred[tuple(voxel)] == 2


def test_oblique_ct_display_has_cardinal_planes(tmp_path):
    angle = .14; affine = np.array([[np.cos(angle),-np.sin(angle),0,0],[np.sin(angle),np.cos(angle),0,0],[0,0,1.4,0],[0,0,0,1]])
    ct = np.zeros((10,11,12),np.float32);gt=np.zeros(ct.shape,np.uint8);gt[3:6,4:7,5:9]=2
    paths = {k:save(tmp_path/(k+'.nii.gz'),ct if k=='ct' else gt,affine) for k in ('ct','ground_truth','prediction')}
    volume=load_volume('case',paths)
    assert np.allclose(volume.shared.reference.affine[:3,:3],np.diag(volume.shared.reference.header.get_zooms()[:3]))
    assert volume.metrics['plaque']['dice'] == 1
    assert volume.warnings


def test_qform_sform_conflict_is_not_silently_hidden(tiny):
    image=nib.load(tiny['ct']);a=image.affine.copy();a[0,3]+=10
    image.set_qform(a,1);nib.save(image,tiny['ct'])
    with pytest.raises(ValueError,match='qform/sform'): checked_image(tiny['ct'])


def test_slice_conventions_and_click_coordinates():
    a=np.arange(4*5*6).reshape(4,5,6);pos=[1,2,3]
    assert slice_array(a,pos,'Axial')[2,2]==a[1,2,3]
    assert slice_array(a,pos,'Coronal')[2,2]==a[1,2,3]
    assert slice_array(a,pos,'Sagittal')[2,2]==a[1,2,3]
    for plane in ('Axial','Coronal','Sagittal'):
        assert screen_to_indices(2,2,a.shape,pos,plane)==pos


def test_binary_plaque_task_is_displayed_as_plaque(tiny):
    image=nib.load(tiny['prediction']);data=(image.get_fdata()==2).astype(np.uint8);save(tiny['prediction'],data,image.affine)
    v=load_volume('case',tiny,task='plaque');assert v.metrics['plaque']['dice']==1
    assert 'lumen' not in v.metrics


def test_case_packages_reclassify_and_preserve_originals(tmp_path,tiny):
    original={k:Path(v).read_bytes() for k,v in tiny.items()}
    store=ReviewStore(tmp_path/'reviews')
    first=store.decide('model/one','case','approved',tiny,notes='good')
    assert (store.root/first['folder']/'ct.nii.gz').is_file()
    second=store.decide('model/one','case','needs_review',tiny,reason='boundary')
    assert not (store.root/first['folder']).exists()
    assert (store.root/second['folder']/'prediction.nii.gz').is_file()
    other=store.decide('model/two','case','rejected',tiny)
    assert len(store.records())==2
    assert len((store.root/'history.jsonl').read_text().splitlines())==3
    assert all(Path(tiny[k]).read_bytes()==value for k,value in original.items())
    assert (store.root/'reviews.csv').is_file()
    # Review packages are never hard-linked to source files.
    assert os.stat(store.root/other['folder']/'ct.nii.gz').st_ino != os.stat(tiny['ct']).st_ino


def test_failed_export_keeps_previous_decision(tmp_path,tiny,monkeypatch):
    import plaque_review_core as core
    store=ReviewStore(tmp_path/'reviews');previous=store.decide('model','case','approved',tiny)
    def fail(*a,**k): raise OSError('simulated full disk')
    monkeypatch.setattr(core.shutil,'copy2',fail)
    changed=save(tmp_path/'changed.nii.gz',np.zeros((2,2,2),np.uint8));tiny['prediction']=changed
    with pytest.raises(OSError):store.decide('model','case','rejected',tiny)
    assert store.records()['model::case']['category']=='approved'
    assert (store.root/previous['folder']).is_dir()
    assert not list((store.root/'.staging').iterdir())


def test_missing_prediction_can_be_flagged_without_dropping_case(tmp_path,tiny):
    tiny['prediction'].unlink();store=ReviewStore(tmp_path/'reviews')
    record=store.decide('model','case','needs_review',tiny)
    assert record['missing']==['prediction']
    assert (store.root/record['folder']/'review.json').is_file()


def test_discovery_rejects_ambiguous_fingerprints(tmp_path):
    manifest,preds=make_demo(tmp_path/'demo');m=read_json(manifest)
    sets=discover_predictions(tmp_path,m,predictions=preds);assert len(sets)==2
    for key in ('one','two'):
        root=tmp_path/'runner'/'run'/'predictions'/'model'/key;root.mkdir(parents=True)
        (root/'run.json').write_text(json.dumps(dict(split_sha256='synthetic-demo',task='multiclass',partition='val')))
    with pytest.raises(ValueError,match='fingerprints'):discover_predictions(tmp_path,m,runs=['run'])


def test_component_filter_is_in_mm3_and_uses_26_connectivity():
    score=np.zeros((6,6,6));score[1,1,1]=score[2,2,2]=score[4,4,4]=.8
    spec=dict(threshold=.5,min_volume_mm3=1.)
    mask=binary_candidate(score,np.ones_like(score,bool),spec,.6)
    assert mask.sum()==2 and mask[1,1,1] and not mask[4,4,4]
    assert len(candidate_specs())==33


def test_patient_bootstrap_is_paired_and_handles_undefined_values():
    delta,lo,hi,n=paired_interval({'a':.6,'b':.8},{'a':.5,'b':.7})
    assert n==2 and np.isclose(delta,.1) and np.isclose(lo,.1) and np.isclose(hi,.1)


@pytest.fixture(scope='session')
def app():
    from PySide6 import QtWidgets
    instance=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield instance


def until(app,predicate,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents()
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('GUI operation did not finish')


def test_viewer_renders_synchronizes_hotkeys_and_exports(app,tmp_path):
    from PySide6 import QtCore,QtTest
    from review_plaque_cases import Reviewer,parser
    args=parser().parse_args(['--out',str(tmp_path/'reviews')]);args.dataset,args.prediction=make_demo(tmp_path/'demo')
    w=Reviewer(args);w.show();until(app,lambda:not w.loading);app.processEvents()
    assert w.volume is not None
    for plane in ('Axial','Coronal','Sagittal'):
        left,right=w.canvases[plane,0],w.canvases[plane,1]
        assert left.ct.scene() is not None and left.ct.qimage is not None
        assert np.array_equal(left.ct.image,right.ct.image)
        old=w.indices[{'Axial':2,'Coronal':1,'Sagittal':0}[plane]]
        right.stepped.emit(plane,1);app.processEvents()
        assert w.indices[{'Axial':2,'Coronal':1,'Sagittal':0}[plane]]==old+1
        assert np.array_equal(left.ct.image,right.ct.image)
    w.table.setFocus()
    QtTest.QTest.keyClick(w.table,QtCore.Qt.Key.Key_A,QtCore.Qt.KeyboardModifier.ControlModifier)
    assert not w.exporting and not w.store.records()
    QtTest.QTest.keyClick(w.table,QtCore.Qt.Key.Key_G);assert not w.gt_toggle.isChecked() and w.pred_toggle.isChecked()
    QtTest.QTest.keyClick(w.table,QtCore.Qt.Key.Key_P);assert not w.pred_toggle.isChecked()
    QtTest.QTest.keyClick(w.table,QtCore.Qt.Key.Key_Space);assert w.pred_toggle.isChecked() and w.gt_toggle.isChecked()
    # Typing a reason must not trigger approval hotkeys.
    w.notes.setFocus();QtTest.QTest.keyClicks(w.notes,'a review');assert w.notes.text()=='a review' and not w.exporting
    w.table.setFocus();QtTest.QTest.keyClick(w.table,QtCore.Qt.Key.Key_A)
    until(app,lambda:not w.exporting and not w.loading)
    assert w.store.records()['demo_good::DEMO001']['category']=='approved'
    assert w.case=='DEMO002'
    # Selecting another model keeps the same CT array and linked position.
    shared=w.shared;pos=w.indices.copy();w.model.setCurrentIndex(1);until(app,lambda:not w.loading)
    assert w.shared is shared and w.indices==pos
    assert w.volume.metrics['plaque']['dice']<1
    w.close();app.processEvents()


def benchmark_repo():
    repo=Path(os.environ.get('PLAQUE_TEST_REPO',Path(__file__).resolve().parents[1]))
    if not (repo/'plaque_benchmark').is_dir():pytest.skip('Set PLAQUE_TEST_REPO to an ImageCAS-X checkout for integration tests')
    sys.path.insert(0,str(repo));return repo


def test_full_cpu_validation_sweep_export_and_restart(tmp_path,capsys):
    benchmark_repo()
    from argparse import Namespace
    from plaque_benchmark.prediction import save_prediction
    from run_PlaquePilot_followups import validation_experiments
    shape=(8,9,10);affine=np.diag([.6,.8,1.2,1]);cases={}
    sources={model:tmp_path/model for model in ('nnunet','nnunet_resenc','ffr_unet')}
    for index in (1,2):
        case=f'SHK{index:04d}';gt=np.zeros(shape,np.uint8);gt[1:4,1:4,1:4]=1
        if index==1:gt[2:4,2:4,2:4]=2
        label=save(tmp_path/(case+'_gt.nii.gz'),gt,affine);ct=save(tmp_path/(case+'_ct.nii.gz'),np.zeros(shape,np.float32),affine)
        lad=save(tmp_path/(case+'_lad.nii.gz'),gt,affine)
        lcx_array=np.zeros(shape,np.uint8);lcx_array[6,7,8]=1
        lcx=save(tmp_path/(case+'_lcx.nii.gz'),lcx_array,affine);rca=save(tmp_path/(case+'_rca.nii.gz'),np.zeros(shape,np.uint8),affine)
        cases[case]=dict(partition='val',image=str(ct),label=str(label),branches=dict(LAD=str(lad),LCX=str(lcx),RCA=str(rca)))
        for model,root in sources.items():
            root.mkdir(exist_ok=True)
            score=np.zeros(shape,np.float32);score[2:4,2:4,2:4]=.8;score[6,7,8]=.55 if model!='ffr_unet' else .6
            probability=np.stack([1-score,np.zeros(shape,np.float32),score])
            save_prediction(root,case,nib.load(ct),probability)
            (root/'run.json').write_text(json.dumps(dict(task='multiclass',partition='val',split_sha256='test',model=model,geometry='native')))
    dataset=tmp_path/'dataset.json';manifest=dict(split_sha256='test',cases=cases);dataset.write_text(json.dumps(manifest))
    args=Namespace(bench=tmp_path,dataset=dataset,out=tmp_path/'out',runs=None,prediction=[f'{k}={v}' for k,v in sources.items()],export_probabilities=False)
    args.out.mkdir();validation_experiments(args,manifest)
    candidates=read_json(args.out/'validation_candidates.json')
    assert len(candidates)==33 and candidates[0]['plaque_dice']==1 and candidates[0]['patients']==2 and candidates[0]['positive_patients']==1
    # The empty reference is never scored as perfect Dice.
    cache=list((args.out/'case_cache').glob('*.json'))
    rows=[r for path in cache for r in read_json(path)]
    assert all(row['dice'] is None for row in rows if row['case']=='SHK0002')
    selected=read_json(args.out/'provisional_winner.json')['spec']
    assert selected['threshold']==.65 or selected['min_volume_mm3']==1
    root=args.out/'provisional_predictions';assert read_json(root/'run.json')['label_only']
    v=load_volume('SHK0001',dict(ct=Path(cases['SHK0001']['image']),ground_truth=Path(cases['SHK0001']['label']),prediction=root/'SHK0001.nii.gz'),task='plaque')
    assert v.metrics['plaque']['dice']==1
    identity=(args.out/'validation_identity.json').read_bytes()
    validation_experiments(args,manifest)
    assert '[cached]' in capsys.readouterr().out and (args.out/'validation_identity.json').read_bytes()==identity


def test_paired_ablation_keeps_settings_and_reuses_completed_stages(tmp_path,monkeypatch):
    benchmark_repo()
    import plaque_benchmark.reporting as reporting
    import run_PlaquePilot_followups as followup
    from argparse import Namespace
    repo=tmp_path/'repo';repo.mkdir()
    for name in ('plaque_benchmark','models','utils'):(repo/name).mkdir()
    for name in ('run_PlaquePilot.py','run_plaque_benchmark.sh'):(repo/name).write_text('# dummy\n')
    bench=repo/'experiments'/'bench';run=bench/'runner'/'on';configs={}
    for model in ('ffr_unet','nnunet'):
        root=run/'predictions'/model/'checkpoint';root.mkdir(parents=True)
        (root/'run.json').write_text(json.dumps(dict(task='multiclass',partition='val',split_sha256='test')))
        cfg=dict(epochs=80,steps=100,batch_size=2,patch=[64,64,64],augmentation={'step':.1})
        configs[model]=cfg
        training=run/'training'/model;training.mkdir(parents=True)
        (training/'override.json').write_text(json.dumps(cfg))
        (training/'specification.json').write_text(json.dumps(dict(context=dict(configuration=cfg))))
        report=run/'reports'/model;report.mkdir(parents=True)
        for name,data in [('summary.json',dict(run={})),('metrics.json',[]),('lesions.json',[])]: (report/name).write_text(json.dumps(data))
    dataset=repo/'dataset.json';dataset.write_text(json.dumps(dict(split_sha256='test',cases={})))
    args=Namespace(repo=repo,bench=bench,dataset=dataset,runs=['on'],prediction=None,ablation_models=['ffr_unet','nnunet'],out=repo/'out',epochs=80,steps=100,ablation_prefix='off')
    args.out.mkdir();commands=[]
    class FakeProcess:
        def __init__(self,command,**kwargs):
            commands.append(command);mode=command[2];model=command[command.index('--models')+1]
            root=bench/'runner'/'off_baseline25'
            if mode=='baseline-init':
                folder=root/'init'/model;folder.mkdir(parents=True,exist_ok=True)
                cfg=dict(configs[model],augmentation=None)
                (folder/'context.json').write_text(json.dumps(dict(configuration=cfg)))
                (folder/'PASS.json').write_text(json.dumps(dict(context=str(folder/'context.json'))))
            else:
                folder=root/'reports'/model;folder.mkdir(parents=True,exist_ok=True)
                for name,data in [('summary.json',dict(run={})),('metrics.json',[]),('lesions.json',[])]: (folder/name).write_text(json.dumps(data))
            self.stdout=['fake training output\n']
        def wait(self,**kwargs):return 0
    monkeypatch.setattr(followup.subprocess,'Popen',FakeProcess)
    compared=[];monkeypatch.setattr(reporting,'compare',lambda reports,out:compared.append((reports,out)))
    followup.paired_ablation(args)
    assert len(commands)==4 and len(compared[0][0])==4
    assert read_json(args.out/'ctta_off_configs'/'ffr_unet.json')['augmentation'] is None
    followup.paired_ablation(args)
    assert len(commands)==4 # Completed init and train stages are reused.


def test_reclassification_rolls_back_if_index_write_fails(tmp_path,tiny,monkeypatch):
    import plaque_review_core as core
    store=ReviewStore(tmp_path/'reviews');previous=store.decide('model','case','approved',tiny)
    original=core.atomic_json
    def failure(path,data):
        if Path(path).name=='reviews.json':raise OSError('simulated index failure')
        return original(path,data)
    monkeypatch.setattr(core,'atomic_json',failure)
    with pytest.raises(OSError):store.decide('model','case','rejected',tiny)
    assert store.records()['model::case']['category']=='approved'
    assert (store.root/previous['folder']/'ct.nii.gz').is_file()
    assert not (store.root/'rejected'/Path(previous['folder']).parts[1]/Path(previous['folder']).parts[2]).exists()


def test_same_category_replace_restores_package_on_install_failure(tmp_path,tiny,monkeypatch):
    import plaque_review_core as core
    store=ReviewStore(tmp_path/'reviews');previous=store.decide('model','case','approved',tiny,notes='original')
    original=core.os.replace;target=store.root/previous['folder']
    def failure(source,dest):
        if Path(dest)==target and not str(source).endswith('.backup'):raise OSError('simulated rename failure')
        return original(source,dest)
    monkeypatch.setattr(core.os,'replace',failure)
    with pytest.raises(OSError):store.decide('model','case','approved',tiny,notes='new')
    assert read_json(target/'review.json')['notes']=='original'
    assert store.records()['model::case']['notes']=='original'


def test_locked_csv_does_not_lose_saved_review(tmp_path,tiny,monkeypatch):
    import plaque_review_core as core
    store=ReviewStore(tmp_path/'reviews');original=core.os.replace
    def failure(source,dest):
        if Path(dest).name=='reviews.csv':raise PermissionError('CSV open in Excel')
        return original(source,dest)
    monkeypatch.setattr(core.os,'replace',failure)
    record=store.decide('model','case','approved',tiny)
    assert record['export_warnings'] and store.records()['model::case']['category']=='approved'

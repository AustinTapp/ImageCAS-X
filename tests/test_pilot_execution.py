import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
import nibabel as nib
import pytest

MODULE=Path(__file__).resolve().parents[1]/'run_PlaquePilot.py'
spec=importlib.util.spec_from_file_location('pilot_execution',MODULE)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
from plaque_benchmark.synthetic import create
from plaque_benchmark.data import read_json,file_sha256


@pytest.fixture
def source(tmp_path):
    return create(tmp_path/'source',cases=6)


def test_all_accepted_preserving_voxels_and_sources(source,tmp_path):
    images,masks=source
    original=masks/'SHK0001/LAD.nii.gz';im=nib.load(original)
    # A physically identical annotation with a reversed voxel axis.
    matrix=np.eye(4);matrix[0,0]=-1;matrix[0,3]=im.shape[0]-1
    wanted=np.asanyarray(im.dataobj).copy()
    nib.save(nib.Nifti1Image(wanted[::-1],im.affine@matrix),original)
    empty=masks/'SHK0002/LCX.nii.gz';nib.save(nib.Nifti1Image(np.zeros(im.shape,np.uint8),im.affine),empty)
    files=list(images.rglob('*.nii.gz'))+list(masks.rglob('*.nii.gz'))
    hashes={p:file_sha256(p) for p in files}
    bench=tmp_path/'bench';manifest=pilot.prepare_confirmed(images,masks,bench)
    assert len(manifest['cases'])==6
    assert all(row['annotation_status']=='accept' for row in manifest['cases'].values())
    assert np.array_equal(np.asanyarray(nib.load(manifest['cases']['SHK0001']['branches']['LAD']).dataobj),wanted)
    assert all(file_sha256(p)==h for p,h in hashes.items())
    again=pilot.prepare_confirmed(images,masks,bench);assert again==manifest
    assert any('empty' in flag for flag in manifest['cases']['SHK0002']['original_audit_flags'])


def test_changed_source_aborts_locked_cohort(source,tmp_path):
    images,masks=source;bench=tmp_path/'bench';pilot.prepare_confirmed(images,masks,bench)
    path=masks/'SHK0001/LAD.nii.gz';im=nib.load(path);data=np.asanyarray(im.dataobj).copy();data[0,0,0]=2
    nib.save(nib.Nifti1Image(data,im.affine),path)
    with pytest.raises(ValueError,match='inputs or split policy changed'):
        pilot.prepare_confirmed(images,masks,bench)


def test_existing_split_preserved_and_different_cohort_rejected(source,tmp_path):
    images,masks=source
    pilot.prepare_confirmed(images,masks,tmp_path/'old')
    old=tmp_path/'old/split.json';parts=pilot.read(old)['partitions']
    pilot.prepare_confirmed(images,masks,tmp_path/'new',reuse_split=old)
    assert pilot.read(tmp_path/'new/split.json')['partitions']==parts
    changed=pilot.read(old);changed['partitions']['train'].append('SHK9999');pilot.write(tmp_path/'different.json',changed)
    with pytest.raises(ValueError,match='different cohort'):
        pilot.prepare_confirmed(images,masks,tmp_path/'bad',reuse_split=tmp_path/'different.json')
    assert not (tmp_path/'bad/prepared/dataset.json').exists()


def test_unsupported_labels_fail_without_subset_manifest(source,tmp_path):
    images,masks=source;path=masks/'SHK0001/LAD.nii.gz';im=nib.load(path)
    data=np.asanyarray(im.dataobj).copy();data[0,0,0]=3;nib.save(nib.Nifti1Image(data,im.affine),path)
    with pytest.raises(ValueError,match='No partial cohort'):
        pilot.prepare_confirmed(images,masks,tmp_path/'bench')
    assert not (tmp_path/'bench/prepared/dataset.json').exists()


def test_partial_prepare_resumes_completed_cases(source,tmp_path,monkeypatch):
    images,masks=source;path=masks/'SHK0006/LAD.nii.gz';im=nib.load(path)
    original=np.asanyarray(im.dataobj).copy()
    # Simulate interruption after the last case is written: no approved manifest yet.
    real=pilot.write
    def interrupt(path,value):
        real(path,value)
        if str(path).endswith('case_receipts/SHK0006.json'):
            raise RuntimeError('interrupted')
    monkeypatch.setattr(pilot,'write',interrupt)
    with pytest.raises(RuntimeError):pilot.prepare_confirmed(images,masks,tmp_path/'bench')
    monkeypatch.setattr(pilot,'write',real)
    # All completed case receipts must be reusable without any new NIfTI saves.
    import plaque_benchmark.data as data
    monkeypatch.setattr(data,'save_image',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected rewrite')))
    assert len(pilot.prepare_confirmed(images,masks,tmp_path/'bench')['cases'])==6


def test_nn_id_allocation_and_persisted_environment(source,tmp_path,monkeypatch):
    images,masks=source;bench=tmp_path/'bench';manifest=pilot.prepare_confirmed(images,masks,bench)
    for key in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results'):
        monkeypatch.setenv(key,str(tmp_path/key))
    (tmp_path/'nnUNet_raw/Dataset710_Existing').mkdir(parents=True)
    state=pilot.nn_settings(bench,tmp_path,manifest)
    assert state['dataset_id']==711
    monkeypatch.setenv('nnUNet_raw','different')
    assert pilot.nn_settings(bench,tmp_path,manifest)==state
    assert pilot.os.environ['nnUNet_raw']==str(tmp_path/'nnUNet_raw')


def test_empty_branch_evaluation_is_explicit_not_healthy(source,tmp_path):
    from plaque_benchmark.evaluation import branch_assignment,evaluate
    from plaque_benchmark.prediction import run_metadata,save_prediction
    images,masks=source
    path=masks/'SHK0001/LAD.nii.gz';im=nib.load(path)
    nib.save(nib.Nifti1Image(np.zeros(im.shape,np.uint8),im.affine),path)
    manifest=pilot.prepare_confirmed(images,masks,tmp_path/'bench')
    # Evaluate that case in a small deliberately constructed validation manifest.
    one={'schema':1,'split_sha256':manifest['split_sha256'],'cases':{'SHK0001':dict(manifest['cases']['SHK0001'],partition='val')}}
    pred=tmp_path/'pred';ref=nib.load(one['cases']['SHK0001']['image'])
    probs=np.zeros((3,*ref.shape),np.float32);probs[0]=1;probs[:,4:6,4:6,4:6]=0;probs[1,4:6,4:6,4:6]=1
    run_metadata(pred,one,'val','multiclass');save_prediction(pred,'SHK0001',ref,probs)
    report=evaluate(one,'val',pred,tmp_path/'report')
    assert report['unevaluable_empty_reference_branches']==[dict(case='SHK0001',branch='LAD')]
    assert report['cases']==1
    assert not any(r['branch']=='LAD' for r in read_json(tmp_path/'report/metrics.json'))
    assigned,coords=branch_assignment(np.ones((2,2,2),bool),{b:np.zeros((2,2,2)) for b in ('LAD','LCX','RCA')},np.eye(4))
    assert np.all(assigned==-1)


def test_native_init_then_train_end_to_end(source,tmp_path,monkeypatch):
    # The real orchestration, gates, child training, inference and evaluation.
    # CPU/tiny settings only; production defaults remain A5000/100 updates.
    import torch
    torch.set_num_threads(1)
    repo=Path(__import__('plaque_benchmark').__file__).resolve().parent.parent
    monkeypatch.chdir(repo)
    images,masks=source
    config_dir=tmp_path/'configs';config_dir.mkdir()
    pilot.write(config_dir/'ffr_unet.json',dict(patch=[16]*3,batch_size=1,accumulate=1,workers=0,
        cache_build_workers=2,model_params={'base_filters':4},validation_patches=2,patience=0))
    bench=tmp_path/'bench'
    args=['--repo',str(repo),'--bench',str(bench),'--images',str(images),'--masks',str(masks),
          '--models','ffr_unet','--config-dir',str(config_dir),'--epochs','2','--steps','1','--init-steps','1',
          '--device','cpu','--allow-other-env']
    assert pilot.main(['init',*args])==0
    run=bench/'runner/PlaquePilot_ctta25'
    assert (run/'init/ffr_unet/PASS.json').is_file()
    assert not (run/'training').exists()  # Initialization must stop here.
    assert pilot.main(['train',*args])==0
    state=torch.load(run/'training/ffr_unet/model/last.pt',weights_only=True)
    assert state['epoch']==2 and state['history'][-1]['optimizer_steps']==2
    assert (run/'reports/ffr_unet/summary.json').is_file()
    assert (run/'comparison').is_dir()


def test_preprocessing_reused_and_foreign_metadata_not_overwritten():
    class Fake:
        def __init__(self):self.ready=False;self.prepared=0;self.seen=[]
        def nn_configuration(self,model):
            self.seen.append(model)
            if not self.ready:raise FileNotFoundError('plan absent')
        def prepare_nnunet(self):self.prepared+=1;self.ready=True
    runner=Fake();models=['cas_net','nnunet','nnunet_resenc']
    pilot.ensure_preprocessed(runner,models);pilot.ensure_preprocessed(runner,models)
    assert runner.prepared==1
    runner.nn_configuration=lambda model:(_ for _ in ()).throw(ValueError('foreign split'))
    with pytest.raises(ValueError,match='foreign split'):pilot.ensure_preprocessed(runner,models)
    assert runner.prepared==1

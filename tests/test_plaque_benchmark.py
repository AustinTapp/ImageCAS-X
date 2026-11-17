from copy import deepcopy
from pathlib import Path
import pickle

import nibabel as nib
import numpy as np
import pytest

from plaque_benchmark.data import BRANCHES, audit, discover, geometry, label_array, make_split, prepare, read_json, save_image, select, target_array, write_json
from plaque_benchmark.evaluation import evaluate, lesions, overlap, surfaces
from plaque_benchmark.fusion import fuse_arrays
from plaque_benchmark.geometry import order_endpoints, resample_path, skeleton_endpoints
from plaque_benchmark.prediction import read_prediction, run_metadata, save_prediction
from plaque_benchmark.smpr import reconstruct, splat, straight_grid, transport_frames


@pytest.fixture
def dataset(tmp_path):
    from plaque_benchmark.synthetic import create
    images,masks = create(tmp_path/"synthetic")
    report = audit(images,masks)
    assert all(row["status"] == "eligible" for row in report["cases"])
    split = make_split(report)
    manifest = prepare(report,split,tmp_path/"prepared")
    return images,masks,report,split,manifest


def test_discovery_qa_split_targets(dataset):
    images,masks,report,split,manifest = dataset
    assert len(discover(images,masks)) == 9
    assert make_split(report) == split
    ids = [case for group in split["partitions"].values() for case in group]
    assert len(ids) == len(set(ids)) == 9
    row = next(iter(manifest["cases"].values()))
    labels = target_array(row,"multiclass")
    assert set(np.unique(labels)) == {0,1,2}
    assert np.array_equal(target_array(row,"lumen"),labels == 1)
    assert np.array_equal(target_array(row,"plaque"),labels == 2)
    assert set(np.unique(target_array(row,"branch_lumen"))) == {0,1,2,3}
    assert any(b["plaque_free"] for c in report["cases"] for b in c["branches"].values())


def test_110_split_sizes():
    report = {"cases":[{"case":f"SHK{i:04d}","status":"eligible","review":"accept",
              "branches":{b:{"plaque_voxels":(i+j)%2} for j,b in enumerate(BRANCHES)}} for i in range(110)]}
    split = make_split(report)
    assert [len(x) for x in split["partitions"].values()] == [77,16,17]


def test_alignment_and_ambiguity(dataset):
    images,masks,*_ = dataset
    path = masks/"SHK0001/LAD.nii.gz"
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0,3] += 100
    save_image(path,np.asanyarray(image.dataobj),affine)
    case = audit(images,masks)["cases"][0]
    assert case["status"] == "excluded" and any("grid mismatch" in e for e in case["errors"])
    original = images/"SHK0001/series_random.nii.gz"
    save_image(original.with_name("second.nii.gz"),np.zeros((32,32,32),np.int16),image.affine)
    assert "ambiguous_ct" in audit(images,masks)["cases"][0]["errors"]


def test_empty_negative_and_labels(dataset):
    images,masks,*_ = dataset
    path = masks/"SHK0001/LAD.nii.gz"
    image = nib.load(path)
    save_image(path,np.zeros(image.shape,np.uint8),image.affine)
    assert "empty annotation" in " ".join(audit(images,masks)["cases"][0]["errors"])
    with pytest.raises(ValueError):
        label_array(nib.Nifti1Image(np.full((4,4,4),3,np.uint8),np.eye(4)))


def test_qform_sform():
    image = nib.Nifti1Image(np.zeros((5,5,5)),np.eye(4))
    affine = np.eye(4)
    affine[1,3] = 3
    image.set_qform(affine,code=1)
    with pytest.raises(ValueError,match="disagree"):
        geometry(image)
    image.set_qform(np.eye(4),code=0)
    image.set_sform(np.eye(4),code=0)
    with pytest.raises(ValueError,match="Neither"):
        geometry(image)


def test_locked_metadata(tmp_path):
    path = tmp_path/"split.json"
    write_json(path,{"seed":2028},lock=True)
    write_json(path,{"seed":2028},lock=True)
    with pytest.raises(ValueError,match="Locked"):
        write_json(path,{"seed":2029},lock=True)


def test_resampling_frames():
    points = np.array([[0,0,0],[0,0,0],[0,0,3],[0,0,7]],float)
    new,arc,radii = resample_path(points,.4,[2,2,2,1])
    assert np.diff(arc).max() <= .400001
    assert np.allclose(new[[0,-1]],points[[0,-1]]) and np.all(radii > 0)
    tangent,normal,binormal = transport_frames(new)
    assert np.allclose((tangent*normal).sum(1),0)
    assert np.allclose(np.cross(tangent,normal),binormal)
    assert np.allclose(np.linalg.norm(normal,axis=1),1)


def test_aorta_free_roots():
    points = np.array([[1,2,3],[1,2,20.]])
    with pytest.raises(ValueError,match="Ambiguous"):
        order_endpoints(points,np.array([2.,2.]))
    ordered,confidence = order_endpoints(points,np.array([2.,2.]),points[-1])
    assert np.array_equal(ordered[0],points[-1]) and confidence["method"] == "manual_ras"
    with pytest.raises(ValueError,match="too far"):
        order_endpoints(points,np.array([2.,2.]),[100,100,100])
    x,y,z = np.indices((23,23,45))
    mask = ((x-11)**2+(y-11)**2 < (4.3-.05*z)**2) & (z >= 5) & (z <= 39)
    ends,widths = skeleton_endpoints(mask,np.eye(4))
    ordered,confidence = order_endpoints(ends,widths)
    assert ordered[0,2] < ordered[-1,2] and confidence["caliber_ratio"] > 1.2


def test_smpr_physical_roundtrip():
    shape = (32,32,40)
    affine = np.diag([.7,.8,.9,1.])
    affine[:3,3] = [-20,35,10]
    x,y,z = np.indices(shape)
    ct = (x+2*y+3*z).astype(np.float32)
    lumen = ((x-16)**2+(y-16)**2 <= 9) & (z >= 4) & (z <= 34)
    seg = lumen.astype(np.uint8)
    seg[(x == 19) & (y == 16) & (z >= 12) & (z <= 15)] = 2
    points,arc,_ = resample_path(nib.affines.apply_affine(affine,[[16,16,4],[16,16,34]]),.3)
    grid,_ = straight_grid(points,arc,.3,8.)
    image,labels,native,valid = reconstruct(ct,affine,grid,seg)
    expected = native[...,0]+2*native[...,1]+3*native[...,2]
    assert np.allclose(image[valid],expected[valid],atol=1e-4)
    assert (labels == 2).any()
    total,weights = np.zeros(shape,np.float32),np.zeros(shape,np.float32)
    splat(total,weights,native,(labels > 0).astype(float),valid)
    pred = total/np.maximum(weights,1e-6) >= .5
    assert overlap(pred,lumen,.7*.8*.9)["dice"] > .9
    center = grid.shape[0]//2
    assert np.allclose(nib.affines.apply_affine(affine,native[center,center]),points,atol=2e-6)
    assert not (weights[0] > 0).any()


def test_metrics_empty_and_lesion_matching():
    empty = np.zeros((14,14,14),bool)
    assert overlap(empty,empty,1)["dice"] is None
    gt,pred = empty.copy(),empty.copy()
    gt[3:6,3:6,3:6] = True
    gt[8:11,3:6,3:6] = True
    pred[3:11,3:6,3:6] = True
    stats,details = lesions(pred,gt,.5)
    assert stats["lesion_tp"] == 1 and stats["lesion_fn"] == 1 and stats["lesion_merge_ambiguities"] == 1
    assert len(details) == 2
    stats = surfaces(gt,gt,(.6,.7,.9),1.)
    assert stats["hd95_mm"] == 0 and stats["surface_dice"] == 1
    assert surfaces(empty,gt,(1,1,1),1)["surface_status"] == "empty_prediction"


def test_fusion_coverage():
    scores = np.array([[.2,.8,0],[.8,0,0]])
    coverage = np.array([[1,1,0],[1,0,0]],bool)
    fused,support = fuse_arrays(scores,coverage)
    assert np.allclose(fused,[.5,.8,0]) and support.tolist() == [True,True,False]
    with pytest.raises(ValueError):
        fuse_arrays(scores,coverage,weights=[-1,1])
    _,support = fuse_arrays(scores,coverage,method="learned",coefficients={"coef":[1,1],"intercept":0})
    assert support.tolist() == [True,False,False]


def test_probability_geometry(dataset,tmp_path):
    ref = nib.load(next(iter(dataset[-1]["cases"].values()))["image"])
    probability = np.zeros((3,*ref.shape),np.float32)
    probability[0] = 1
    save_prediction(tmp_path,"a",ref,probability)
    assert read_prediction(tmp_path,"a",ref,"multiclass")[1].all()
    bad = nib.Nifti1Image(np.zeros(ref.shape),ref.affine @ np.diag([2,1,1,1]))
    with pytest.raises(ValueError,match="affine"):
        read_prediction(tmp_path,"a",bad,"multiclass")


def test_protected_test(dataset,tmp_path):
    from plaque_benchmark.protection import freeze, guard
    path = tmp_path/"weights.pt"
    path.write_bytes(b"synthetic")
    frozen = tmp_path/"promotion.json"
    with pytest.raises(ValueError,match="requires"):
        guard(dataset[-1],"test",None)
    record = freeze(dataset[-1],frozen,checkpoints=[path],decision="Synthetic promotion.")
    guard(dataset[-1],"test",frozen,record["candidate_ids"][0])
    with pytest.raises(ValueError,match="not promoted"):
        guard(dataset[-1],"test",frozen,"other")


def test_patch_pickle_losses(dataset):
    import torch
    from plaque_benchmark.native import Patches, objective
    ds = Patches(select(dataset[-1],"train"),"multiclass",[16]*3,4,3,training=True,cache_cases=1,
                 sampling={"plaque":.4,"healthy":.2})
    image,target = pickle.loads(pickle.dumps(ds))[0]
    assert image.shape == (1,16,16,16)
    for name in ("dice_ce","focal","focal_tversky","weighted_ce","boundary","cldice"):
        logits = torch.zeros(1,3,16,16,16,requires_grad=True)
        loss = objective(logits,target[None],{"loss":name})
        assert torch.isfinite(loss)
        loss.backward()


@pytest.mark.parametrize("name,side,params",[("cas_net",32,{}),("ffr_unet",16,{"base_filters":4}),
    ("swin_unetr",64,{"feature_size":12,"depths":[1,1,1,1]})])
def test_multiclass_heads(name,side,params):
    import torch
    from plaque_benchmark.native import build
    model = build(name,"multiclass",params).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1,1,side,side,side))["logits"].shape == (1,3,side,side,side)


def smoke_config():
    from plaque_benchmark.native import config_for
    config = config_for("ffr_unet","smoke")
    config.update(epochs=2,steps=1,patch=[16]*3,batch_size=1,validation_patches=1,model_params={"base_filters":4},
                  sampling={"plaque":.5})
    return config


def test_train_infer_evaluate_resume(dataset,tmp_path):
    import torch
    from plaque_benchmark.native import train,infer
    run = tmp_path/"training"
    config = smoke_config()
    train(dataset[-1],"multiclass",config,run,device="cpu")
    state = torch.load(run/"last.pt",weights_only=True)
    assert state["epoch"] == 2 and state["history"][-1]["optimizer_steps"] == 2
    train(dataset[-1],"multiclass",config,run,device="cpu",resume=True)
    infer(dataset[-1],"val",run/"best.pt",tmp_path/"predictions",device="cpu")
    summary = evaluate(dataset[-1],"val",tmp_path/"predictions",tmp_path/"report")
    assert summary["cases"] == len(select(dataset[-1],"val"))
    assert "plaque" in summary and "lumen" in summary


def test_nnunet_export_import(dataset,tmp_path):
    from plaque_benchmark.external import native_export,install_split,import_nnunet
    manifest = dataset[-1]
    exported = native_export(manifest,tmp_path/"raw")
    assert len(list((exported/"imagesTr").glob("*.nii.gz"))) == len(select(manifest,"train"))+len(select(manifest,"val"))
    assert not (exported/"imagesTs").exists()
    path = install_split(exported,tmp_path/"preprocessed")
    assert read_json(path) == read_json(exported/"splits_final.json")
    scores = tmp_path/"scores"
    for case,row in select(manifest,"val"):
        ref = nib.load(row["image"])
        labels = target_array(row,"multiclass")
        save_image(scores/f"{case}.nii.gz",labels,ref.affine)
        np.savez_compressed(scores/f"{case}.npz",probabilities=np.eye(3,dtype=np.float32)[labels].transpose(3,2,1,0))
    import_nnunet(manifest,"val",scores,tmp_path/"standard","multiclass")
    assert evaluate(manifest,"val",tmp_path/"standard",tmp_path/"report")["plaque"]["dice"] == 1


def test_specialists_ensemble_refinement_plotting(dataset,tmp_path):
    from plaque_benchmark.fusion import specialists,ensemble,refine
    from plaque_benchmark.reporting import compare
    manifest = dataset[-1]
    roots = {"lumen":tmp_path/"lumen","plaque":tmp_path/"plaque"}
    for task,root in roots.items():
        run_metadata(root,manifest,"val",task,model="synthetic_"+task,candidate_id=task)
        for case,row in select(manifest,"val"):
            ref = nib.load(row["image"])
            labels = target_array(row,task)
            save_prediction(root,case,ref,np.eye(2,dtype=np.float32)[labels].transpose(3,0,1,2))
    specialists(manifest,"val",roots["lumen"],roots["plaque"],tmp_path/"specialists")
    assert evaluate(manifest,"val",tmp_path/"specialists",tmp_path/"report1")["lumen"]["dice"] == 1
    ensemble(manifest,"val",[roots["plaque"],tmp_path/"specialists"],tmp_path/"ensemble")
    refine(manifest,"val",tmp_path/"ensemble",tmp_path/"refined",min_volume_mm3=.01,
           lumen_root=roots["lumen"],max_lumen_distance_mm=5)
    assert evaluate(manifest,"val",tmp_path/"refined",tmp_path/"report2")["plaque"]["dice"] == 1
    assert len(compare([tmp_path/"report1",tmp_path/"report2"],tmp_path/"plots")) == 2
    assert (tmp_path/"plots/case_failure_matrix.png").is_file()


def test_inverse_mapping_extraction_failure(dataset,tmp_path):
    from plaque_benchmark.smpr import batch_reconstruct,inverse_batch
    manifest = dataset[-1]
    curves = {"split_sha256":manifest["split_sha256"],"partition":"val","source":"oracle",
              "geometry":"oracle-centerline-assisted","cases":{}}
    for case,row in select(manifest,"val"):
        ref = nib.load(row["image"])
        curves["cases"][case] = {}
        for i,branch in enumerate(BRANCHES):
            if branch == "RCA":
                curves["cases"][case][branch] = {"success":False,"paths":[],"reason":"Synthetic failure"}
                continue
            points,arc,radius = resample_path(nib.affines.apply_affine(ref.affine,[[6+9*i,16,4],[6+9*i,16,27]]),.3,[2,2])
            path = tmp_path/f"{case}_{branch}.npz"
            np.savez_compressed(path,points_ras_mm=points,arc_mm=arc,radius_mm=radius)
            curves["cases"][case][branch] = {"success":True,"paths":[str(path)]}
    index = batch_reconstruct(manifest,curves,tmp_path/"smpr",cross_mm=.3,fov_mm=8,include_labels=True)
    scores = tmp_path/"scores"
    write_json(scores/"run.json",{"candidate_id":"synthetic"})
    for key,item in index["items"].items():
        image = nib.load(item["image"])
        labels = np.asanyarray(nib.load(item["label"]).dataobj)
        save_image(scores/f"{key}.nii.gz",labels,image.affine)
        np.savez_compressed(scores/f"{key}.npz",probabilities=np.eye(3,dtype=np.float32)[labels].transpose(3,0,1,2))
    inverse_batch(manifest,index,scores,tmp_path/"native",engine="xyz",initialization="synthetic")
    assert all(not item["RCA"]["success"] for item in read_json(tmp_path/"native/extraction.json").values())
    result = evaluate(manifest,"val",tmp_path/"native",tmp_path/"report")
    assert result["extraction_success_rate"] == pytest.approx(2/3)
    assert result["plaque"]["dice"] < 1


def test_hard_negative_sampling(dataset,tmp_path):
    from plaque_benchmark.native import Patches
    rows = select(dataset[-1],"train")
    root = tmp_path/"mining"
    run_metadata(root,dataset[-1],"train","plaque",model="synthetic")
    for case,row in rows:
        ref = nib.load(row["image"])
        probs = np.zeros((2,*ref.shape),np.float32)
        probs[0] = 1
        probs[:,6:8,15:17,20:22] = np.array([.1,.9])[:,None,None,None]
        save_prediction(root,case,ref,probs)
    ds = Patches(rows,"multiclass",[16]*3,2,2,training=False,cache_cases=1,
                 sampling={"hard_negative":1.},hard_negative_root=root)
    image,mask = ds[0]
    assert image.shape == (1,16,16,16) and (mask == 1).any()


def test_gate_calibration_exclusion(dataset,tmp_path):
    from plaque_benchmark.fusion import fit_gate,ensemble
    manifest = deepcopy(dataset[-1])
    for row in manifest["cases"].values():
        row["partition"] = "val"
    roots = [tmp_path/"expert1",tmp_path/"expert2"]
    for i,root in enumerate(roots):
        run_metadata(root,manifest,"val","plaque",model=f"synthetic{i}",candidate_id=f"weights{i}")
        for case,row in select(manifest,"val"):
            ref = nib.load(row["image"])
            score = np.where(target_array(row,"plaque"),.85-i*.1,.05+i*.05).astype(np.float32)
            save_prediction(root,case,ref,np.stack([1-score,score]))
    path = tmp_path/"gate.json"
    gate = fit_gate(manifest,roots,path,per_case=32768)
    assert not set(gate["calibration_ids"]) & set(gate["assessment_ids"])
    ensemble(manifest,"val",roots,tmp_path/"gated",method="learned",gating=path)
    assert evaluate(manifest,"val",tmp_path/"gated",tmp_path/"report")["evaluated_ids"] == gate["assessment_ids"]


def test_dynunet_output():
    pytest.importorskip("monai")
    import torch
    from plaque_benchmark.native import build
    model = build("dynunet","multiclass",{"filters":[8,16,32,64]}).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1,1,16,16,16))["logits"].shape == (1,3,16,16,16)


def test_binary_defaults():
    from models.cas_net import CASNet3D
    from models.ffr_unet import FFRUNet3D
    cas,ffr = CASNet3D(),FFRUNet3D(base_filters=4)
    assert cas.msfa.out_conv.out_channels == 2 and ffr.head.out_channels == 1
    cas.load_state_dict(cas.state_dict(),strict=True)
    ffr.load_state_dict(ffr.state_dict(),strict=True)


def test_cldice_empty_and_false_positive():
    import torch
    from plaque_benchmark.losses import cldice
    empty = torch.zeros(1,1,16,16,16)
    line = empty.clone()
    line[:,:,7:9,7:9,2:14] = 1
    assert cldice(line,empty).item() > .9
    assert cldice(empty,empty).item() < 1e-5


def test_interrupted_resume_equivalence(dataset,tmp_path,monkeypatch):
    import torch
    import plaque_benchmark.native as native
    config = smoke_config()
    native.train(dataset[-1],"multiclass",config,tmp_path/"complete",device="cpu")
    original = native._history_files

    def interrupt(out,history):
        raise RuntimeError("Synthetic interruption after completed epoch.")

    monkeypatch.setattr(native,"_history_files",interrupt)
    with pytest.raises(RuntimeError,match="Synthetic interruption"):
        native.train(dataset[-1],"multiclass",config,tmp_path/"interrupted",device="cpu")
    monkeypatch.setattr(native,"_history_files",original)
    native.train(dataset[-1],"multiclass",config,tmp_path/"interrupted",device="cpu",resume=True)
    a = torch.load(tmp_path/"complete/last.pt",weights_only=True)
    b = torch.load(tmp_path/"interrupted/last.pt",weights_only=True)
    assert a["epoch"] == b["epoch"] == 2
    assert all(torch.equal(value,b["model"][key]) for key,value in a["model"].items())

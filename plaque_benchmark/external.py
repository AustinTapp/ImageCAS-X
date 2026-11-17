"""External nnU-Net dataset, trainer, plans and probability interchange."""
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .data import TASKS, digest, file_sha256, label_array, read_json, same_grid, save_image, select, target_array, write_json
from .prediction import run_metadata, save_prediction


def native_export(manifest, raw, dataset_name="Dataset701_SHKPlaque", *, task="multiclass", intensity_offset=0.):
    root = Path(raw)/dataset_name
    if not dataset_name.startswith("Dataset") or len(dataset_name.split("_")[0]) != 10 or not np.isfinite(intensity_offset):
        raise ValueError("Use DatasetNNN_Name and a finite intensity offset.")
    write_json(root/"benchmark_export.json",{"split_sha256":manifest["split_sha256"],"dataset_sha256":digest(manifest),
                                            "task":task,"intensity_offset":intensity_offset},lock=True)
    train,val = [],[]
    names = {"multiclass":{"background":0,"lumen":1,"plaque":2},"lumen":{"background":0,"lumen":1},
             "plaque":{"background":0,"plaque":1},"branch_lumen":{"background":0,"LAD":1,"LCX":2,"RCA":3,"ignore":255}}[task]
    for partition,ids in (("train",train),("val",val)):
        for case,row in select(manifest,partition):
            ids.append(case)
            ref = nib.load(row["image"])
            save_image(root/"imagesTr"/f"{case}_0000.nii.gz",ref.get_fdata(dtype=np.float32)+intensity_offset,ref.affine)
            save_image(root/"labelsTr"/f"{case}.nii.gz",target_array(row,task),ref.affine)
    write_json(root/"dataset.json",{"channel_names":{"0":"CT"},"labels":names,"numTraining":len(train)+len(val),"file_ending":".nii.gz"})
    write_json(root/"splits_final.json",[{"train":train,"val":val}],lock=True)
    return root


def install_split(exported_dataset, preprocessed):
    source = Path(exported_dataset)/"splits_final.json"
    target = Path(preprocessed)/Path(exported_dataset).name/"splits_final.json"
    write_json(target,read_json(source),lock=True)
    return target


def install_trainers(directory=None):
    if directory is None:
        import nnunetv2
        directory = Path(nnunetv2.__path__[0])/"training/nnUNetTrainer/variants/training_length"
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    source = Path(__file__).with_name("nnUNetTrainerSHK.py")
    target = directory/source.name
    if target.is_file() and target.read_bytes() != source.read_bytes():
        raise FileExistsError(f"A different trainer already exists: {target}")
    shutil.copyfile(source,target)
    return target


def smpr_manifest(manifest, indices, out):
    result = {"schema":1,"split_sha256":manifest["split_sha256"],"cases":{}}
    geometry,partitions = None,set()
    for index in indices:
        if index["split_sha256"] != manifest["split_sha256"] or index["partition"] in partitions:
            raise ValueError("sMPR split mismatch or duplicate partition index.")
        partitions.add(index["partition"])
        geometry = geometry or index["geometry"]
        if geometry != index["geometry"]:
            raise ValueError("Mixed geometry supervision in training dataset.")
        for key,row in index["items"].items():
            if row["label"] is None:
                raise ValueError("Reconstruct with --include-labels before preparing sMPR training.")
            result["cases"][key] = {"partition":index["partition"],"image":row["image"],"label":row["label"],
                                    "parent_case":row["case"],"branch":row["branch"]}
    result["geometry"] = geometry
    write_json(Path(out)/"dataset.json",result,lock=True)
    return result


def import_nnunet(manifest, partition, predictions, out, task, *, probability_axes="zyx", checkpoint=None, initialization="external"):
    if probability_axes not in {"zyx","xyz"} or Path(predictions).resolve() == Path(out).resolve():
        raise ValueError("Specify axis order and a separate standardized output directory.")
    identity = file_sha256(checkpoint) if checkpoint else None
    run_metadata(out,manifest,partition,task,manifest.get("geometry","native"),model="external_nnunet",
                 checkpoint_sha256=identity,candidate_id=identity,initialization=initialization)
    for case,row in select(manifest,partition):
        ref = nib.load(row["image"])
        image = nib.load(Path(predictions)/f"{case}.nii.gz")
        if not same_grid(ref,image):
            raise ValueError(f"nnU-Net label grid mismatch: {case}")
        labels = label_array(image,tuple(range(TASKS[task])))
        with np.load(Path(predictions)/f"{case}.npz") as data:
            probs = data["probabilities"]
        if probability_axes == "zyx":
            probs = probs.transpose(0,3,2,1)
        if probs.shape != (TASKS[task],*ref.shape) or not np.isfinite(probs).all() or probs.min() < 0 or not np.allclose(probs.sum(axis=0),1,atol=.003):
            raise ValueError(f"Invalid nnU-Net probabilities: {case}")
        save_prediction(out,case,ref,probs,seg=labels)


def export_inputs(manifest, partition, out, intensity_offset=0.):
    out = Path(out)
    if not np.isfinite(intensity_offset):
        raise ValueError("Invalid intensity offset.")
    write_json(out/"inputs.json",{"split_sha256":manifest["split_sha256"],"partition":partition,"intensity_offset":intensity_offset},lock=True)
    for case,row in select(manifest,partition):
        image = nib.load(row["image"])
        save_image(out/f"{case}_0000.nii.gz",image.get_fdata(dtype=np.float32)+intensity_offset,image.affine)


def transfer_plans(model, exported_dataset, preprocessed, name="MoosaviTransfer"):
    model,exported_dataset = Path(model),Path(exported_dataset)
    source,target = read_json(model/"dataset.json"),read_json(exported_dataset/"dataset.json")
    if len(source["channel_names"]) != len(target["channel_names"]):
        raise ValueError("Transfer requires compatible input channels.")
    plans = read_json(model/"plans.json")
    plans["dataset_name"],plans["plans_name"] = exported_dataset.name,name
    plans["configurations"] = {"3d_fullres":plans["configurations"]["3d_fullres"]}
    plans["configurations"]["3d_fullres"]["data_identifier"] = f"{name}_3d_fullres"
    out = Path(preprocessed)/exported_dataset.name/f"{name}.json"
    write_json(out,plans,lock=True)
    return out

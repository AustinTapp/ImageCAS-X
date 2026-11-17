"""Inspect released Moosavi assets and invoke their nnU-Net checkpoint."""
from pathlib import Path

import nibabel as nib
import numpy as np

from .data import digest, file_sha256, geometry, read_json, write_json


def inspect_assets(model, samples, out):
    model,samples = Path(model),Path(samples)
    plans,dataset = read_json(model/"plans.json"),read_json(model/"dataset.json")
    config = plans["configurations"]["3d_fullres"]
    files = sorted(samples.rglob("*_0000.nii.gz"))
    if not files:
        raise FileNotFoundError("No sample channel-0 NIfTI files.")
    rows = []
    for path in files:
        image = nib.load(path)
        array = image.get_fdata(dtype=np.float32)
        rows.append({"name":path.name,**geometry(image),
                     "intensity_percentiles":np.percentile(array,[0,1,50,99,100]).tolist()})
    report = {"plans_sha256":file_sha256(model/"plans.json"),"dataset":dataset,
        "reader":plans["image_reader_writer"],"spacing_in_reader_order":config["spacing"],
        "patch_in_reader_order":config["patch_size"],"architecture":config.get("architecture",config.get("UNet_class_name")),
        "normalization":config["normalization_schemes"],"training_intensity_properties":plans.get("foreground_intensity_properties_per_channel"),
        "samples":rows,"clinical_label_1":"Non-pathological vessel. Equivalence to patent lumen remains unverified.",
        "geometry_parity":"Unverified. Released arrays lack the native-to-reconstruction deformation.",
        "intensity_offset":"Unverified. Explicitly choose native HU offset before exploratory transfer."}
    write_json(out,report)
    return report


def infer_folder(model, images, out, *, folds=(0,), checkpoint="checkpoint_final.pth", device="cuda", cpu_threads=4, mirror=False):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(cpu_threads)
    predictor = nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=mirror,
        perform_everything_on_device=device.startswith("cuda"),device=torch.device(device),
        verbose=False,verbose_preprocessing=False,allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(str(model),use_folds=tuple(folds),checkpoint_name=checkpoint)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    reader = predictor.plans_manager.image_reader_writer_class()
    files = sorted(Path(images).glob("*_0000.nii.gz"))
    if not files:
        raise ValueError("No inference inputs.")
    checkpoints = [file_sha256(Path(model)/f"fold_{fold}"/checkpoint) for fold in folds]
    stamp = Path(images)/"reconstruction_provenance.json"
    provenance = read_json(stamp) if stamp.is_file() else {}
    identity = digest({"checkpoints":checkpoints,"plans":file_sha256(Path(model)/"plans.json"),
                       "mirror":mirror,"intensity_offset":provenance.get("intensity_offset")})
    write_json(out/"moosavi_inference.json",{"model":str(Path(model).resolve()),"folds":list(folds),"checkpoint":checkpoint,
        "device":device,"mirror":mirror,"geometry_parity":"unverified","model_identity":identity,
        "inputs_sha256":{path.name:file_sha256(path) for path in files},"input_provenance":provenance},lock=True)
    for path in files:
        key = path.name.removesuffix("_0000.nii.gz")
        if (out/f"{key}.nii.gz").is_file() and (out/f"{key}.npz").is_file():
            continue
        image,properties = reader.read_images([str(path)])
        predictor.predict_single_npy_array(image,properties,output_file_truncated=str(out/key),save_or_return_probabilities=True)
    write_json(out/"score_identity.json",{"candidate_id":identity,"checkpoint_sha256":checkpoints,
                                         "model_plans_sha256":file_sha256(Path(model)/"plans.json")})


def prepare_inputs(index, out, intensity_offset):
    from .data import save_image
    if not np.isfinite(intensity_offset):
        raise ValueError("Choose a finite intensity offset.")
    out = Path(out)
    write_json(out/"reconstruction_provenance.json",{"split_sha256":index["split_sha256"],"geometry":index["geometry"],
        "partition":index["partition"],"intensity_offset":intensity_offset,"cross_mm":index["cross_mm"],
        "fov_mm":index["fov_mm"],"long_axis":index["long_axis"],"geometry_parity":"unverified"},lock=True)
    for key,row in index["items"].items():
        image = nib.load(row["image"])
        save_image(out/f"{key}_0000.nii.gz",image.get_fdata(dtype=np.float32)+intensity_offset,image.affine)


def fetch_public_assets(out):
    import gdown
    out = Path(out).resolve()
    for item in read_json(Path(__file__).parent/"configs"/"moosavi_public_assets.json"):
        target = out/item["path"]
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file():
            continue
        temporary = target.with_suffix(target.suffix+".partial")
        result = gdown.download(item["url"],str(temporary),quiet=False,use_cookies=False)
        if result is None:
            raise RuntimeError(f"Public download failed: {target.name}")
        temporary.replace(target)


def verify_public_predictions(model, samples, scores, out):
    from .data import label_array, same_grid
    from .evaluation import overlap, lesions
    scores,samples = Path(scores),Path(samples)
    rows = []
    for path in sorted((samples/"Labels").glob("*.nii.gz")):
        prediction = scores/path.name
        if not prediction.is_file():
            continue
        gt_image,pred_image = nib.load(path),nib.load(prediction)
        if not same_grid(gt_image,pred_image):
            raise ValueError(f"Sample grid mismatch: {path.name}")
        gt,pred = label_array(gt_image) == 2,label_array(pred_image) == 2
        voxel = abs(float(np.linalg.det(gt_image.affine[:3,:3])))
        detection,_ = lesions(pred,gt,voxel)
        rows.append({"sample":path.name,**overlap(pred,gt,voxel),**detection})
    if not rows:
        raise ValueError("No corresponding sample predictions found.")
    record = {"protocol":read_json(scores/"moosavi_inference.json"),"plans_sha256":file_sha256(Path(model)/"plans.json"),
        "global_overlap":"Complete reconstructed vessels, with raw outputs",
        "independence":"Unknown. Public sample results provide engineering verification.",
        "native_SHK_geometry_parity":"unverified","results":rows}
    write_json(out,record)
    return record

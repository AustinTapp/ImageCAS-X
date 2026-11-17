"""Straightened 3D vessel reconstruction with coverage-aware inverse projection."""
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .data import digest, file_sha256, label_array, read_json, same_grid, save_image, select, write_json
from .prediction import run_metadata, save_prediction


def transport_frames(points):
    points = np.asarray(points,float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("Expected an ordered Nx3 polyline.")
    tangents = np.gradient(points,axis=0)
    norm = np.linalg.norm(tangents,axis=1)
    if not np.isfinite(norm).all() or (norm < 1e-8).any():
        raise ValueError("Degenerate centerline tangent.")
    tangents /= norm[:,None]
    normals = np.empty_like(tangents)
    axis = np.eye(3)[np.argmin(np.abs(tangents[0]))]
    normals[0] = axis-axis.dot(tangents[0])*tangents[0]
    normals[0] /= np.linalg.norm(normals[0])
    for i,tangent in enumerate(tangents[1:],1):
        vector = np.cross(tangents[i-1],tangent)
        cosine = tangents[i-1].dot(tangent)
        if cosine < -.95:
            raise ValueError("Abrupt centerline reversal.")
        normal = normals[i-1]
        cross = np.cross(vector,normal)
        normal = normal+cross+np.cross(vector,cross)/(1+cosine)
        normal -= normal.dot(tangent)*tangent
        normals[i] = normal/np.linalg.norm(normal)
    return tangents,normals,np.cross(tangents,normals)


def straight_grid(points, arc, cross_mm=.4, fov_mm=20, long_axis=2, cross_pixels=None):
    if not np.isfinite([cross_mm,fov_mm]).all() or min(cross_mm,fov_mm) <= 0 or long_axis not in (0,1,2):
        raise ValueError("Invalid spacing, field of view, or longitudinal axis.")
    _,normal,binormal = transport_frames(points)
    pixels = cross_pixels if cross_pixels is not None else 2*int(np.ceil(fov_mm/(2*cross_mm)))+1
    if not isinstance(pixels,int) or pixels < 2:
        raise ValueError("Cross-section needs at least two pixels.")
    offsets = (np.arange(pixels)-(pixels-1)/2)*cross_mm
    xyz = (points[None,None]+offsets[:,None,None,None]*normal[None,None]
           +offsets[None,:,None,None]*binormal[None,None])
    affine = np.diag([cross_mm,cross_mm,float(np.diff(arc).mean()),1.])
    affine[:2,3] = offsets[0]
    permutation = [0,1]
    permutation.insert(long_axis,2)
    affine[:3,:3] = affine[:3,permutation]
    return np.moveaxis(xyz,2,long_axis),affine


def reconstruct(ct, affine, xyz, labels=None, outside_hu=-1024):
    native = nib.affines.apply_affine(np.linalg.inv(affine),xyz)
    valid = ((native >= -1e-6) & (native <= np.array(ct.shape)-1+1e-6)).all(axis=-1)
    coordinates = np.moveaxis(np.clip(native,0,np.array(ct.shape)-1),-1,0)
    image = ndi.map_coordinates(ct,coordinates,order=1)
    image[~valid] = outside_hu
    seg = None
    if labels is not None:
        seg = ndi.map_coordinates(labels,coordinates,order=0)
        seg[~valid] = 0
    return image,seg,native.astype(np.float32),valid


def splat(total, weights, native, values, valid):
    if values.shape != valid.shape or native.shape != (*valid.shape,3):
        raise ValueError("Probability/deformation shape mismatch.")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1 or not np.isfinite(native[valid]).all():
        raise ValueError("Invalid probabilities or deformation coordinates.")
    points,probs = native[valid],values[valid]
    shape = np.array(total.shape)
    for start in range(0,len(points),250000):
        point,value = points[start:start+250000],probs[start:start+250000]
        base = np.floor(point).astype(int)
        frac = point-base
        for corner in product((0,1),repeat=3):
            ijk = base+corner
            weight = np.prod(np.where(corner,frac,1-frac),axis=1)
            keep = (weight > 1e-6) & ((ijk >= 0) & (ijk < shape)).all(axis=1)
            index = tuple(ijk[keep].T)
            np.add.at(total,index,weight[keep]*value[keep])
            np.add.at(weights,index,weight[keep])


def overlay(path, ct, label, long_axis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.moveaxis(ct,long_axis,-1)
    mask = None if label is None else np.moveaxis(label,long_axis,-1)
    views = ((slice(None),slice(None),data.shape[2]//2),(data.shape[0]//2,slice(None),slice(None)))
    fig,axes = plt.subplots(2,1,figsize=(8,6))
    for ax,view,title in zip(axes,views,("Cross-section","Longitudinal section")):
        ax.imshow(data[view],cmap="gray",vmin=-150,vmax=650,interpolation="nearest")
        if mask is not None:
            ax.imshow(np.ma.masked_where(mask[view] == 0,mask[view]),cmap="autumn",alpha=.45,vmin=1,vmax=2,interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path,dpi=120)
    plt.close(fig)


def batch_reconstruct(manifest, centerlines, out, cross_mm=.4, fov_mm=20, long_axis=2,
                      overlays=False, include_labels=False, cross_pixels=None):
    if centerlines["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Centerline split mismatch.")
    out = Path(out).resolve()
    old_path = out/"smpr.json"
    cached = read_json(old_path).get("items",{}) if old_path.is_file() else {}
    index = {key:centerlines[key] for key in ("split_sha256","geometry","partition","source")}
    index.update(items={},extraction={},cross_mm=cross_mm,fov_mm=fov_mm,long_axis=long_axis,cross_pixels=cross_pixels)
    index["cross_section_span_mm"] = float((cross_pixels-1)*cross_mm if cross_pixels else 2*np.ceil(fov_mm/(2*cross_mm))*cross_mm)
    for case,row in select(manifest,index["partition"]):
        ref = nib.load(row["image"])
        ct,ct_hash = None,file_sha256(row["image"])
        case_out = out/case
        index["extraction"][case] = {}
        for branch,result in centerlines["cases"][case].items():
            labels = None
            if include_labels and result["paths"]:
                image = nib.load(row["branches"][branch])
                if not same_grid(ref,image):
                    raise ValueError(f"Annotation grid mismatch: {case} {branch}")
                labels = label_array(image)
            count,failures = 0,[]
            for i,path in enumerate(result["paths"]):
                key = f"{case}_{branch}_path{i:02d}"
                signature = digest({"ct":ct_hash,"curve":file_sha256(path),"cross_mm":cross_mm,"fov_mm":fov_mm,
                    "cross_pixels":cross_pixels,"long_axis":long_axis,"labels":file_sha256(row["branches"][branch]) if include_labels else None,
                    "implementation":file_sha256(__file__)})
                prior = cached.get(key,{})
                reusable = (prior.get("signature") == signature and all(Path(prior[name]).is_file() for name in ("image","mapping"))
                            and (not include_labels or prior.get("label") and Path(prior["label"]).is_file())
                            and (not overlays or (case_out/f"{key}_overlay.png").is_file()))
                if reusable:
                    index["items"][key] = prior
                    count += 1
                    continue
                if ct is None:
                    ct = ref.get_fdata(dtype=np.float32)
                with np.load(path) as curve:
                    points,arc = curve["points_ras_mm"],curve["arc_mm"]
                try:
                    xyz,affine = straight_grid(points,arc,cross_mm,fov_mm,long_axis,cross_pixels)
                    image,seg,native,valid = reconstruct(ct,ref.affine,xyz,labels)
                    tangent,_,_ = transport_frames(points)
                    derivative = np.gradient(np.moveaxis(xyz,long_axis,2),axis=2)
                    forward = np.moveaxis((derivative*tangent[None,None]).sum(axis=-1) > 0,2,long_axis)
                    valid &= forward
                    image[~valid] = -1024
                    if seg is not None:
                        seg[~valid] = 0
                    if not valid.any():
                        raise ValueError("Reconstruction has no valid native sampling.")
                except ValueError as exc:
                    failures.append(str(exc))
                    continue
                image_path,label_path,map_path = (case_out/f"{key}_0000.nii.gz",case_out/f"{key}.nii.gz",case_out/f"{key}_mapping.npz")
                save_image(image_path,image,affine)
                if seg is not None:
                    save_image(label_path,seg,affine)
                np.savez_compressed(map_path,native_ijk=native,valid=valid,native_affine=ref.affine,
                                    native_shape=ref.shape,straight_affine=affine)
                if overlays:
                    overlay(case_out/f"{key}_overlay.png",image,seg,long_axis)
                index["items"][key] = {"case":case,"branch":branch,"image":str(image_path),"label":str(label_path) if seg is not None else None,
                    "mapping":str(map_path),"signature":signature,"valid_fraction":float(valid.mean()),
                    "folded_sample_fraction":float((~forward).mean()),"plaque_voxels":None if seg is None else int((seg == 2).sum())}
                count += 1
            index["extraction"][case][branch] = {"success":count > 0,"paths":count,"errors":failures,"centerline_failure":result.get("reason")}
    write_json(out/"smpr.json",index)
    return index


def inverse_batch(manifest, index, scores, out, *, plaque_channel=2, threshold=.5, engine="nnunet", initialization="unspecified"):
    if index["split_sha256"] != manifest["split_sha256"] or not 0 < threshold < 1:
        raise ValueError("Mapping split or threshold mismatch.")
    scores,out = Path(scores),Path(out)
    identity_file = scores/"score_identity.json" if engine == "nnunet" else scores/"run.json"
    identity = read_json(identity_file).get("candidate_id")
    if not identity:
        raise ValueError("Missing scoring-model identity.")
    signature = digest({"model":identity,"threshold":threshold,"plaque_channel":plaque_channel,
        "cross_mm":index["cross_mm"],"cross_pixels":index.get("cross_pixels"),"fov_mm":index["fov_mm"],
        "long_axis":index["long_axis"],"geometry":index["geometry"]})
    run_metadata(out,manifest,index["partition"],"plaque",index["geometry"],model=f"smpr_{engine}",threshold=threshold,
        score_directory=str(scores.resolve()),initialization=initialization,candidate_id=signature)
    by_case = {case:[] for case,_ in select(manifest,index["partition"])}
    for key,item in index["items"].items():
        by_case[item["case"]].append((key,item))
    extraction = index["extraction"]
    for case,row in select(manifest,index["partition"]):
        ref = nib.load(row["image"])
        total,weights = np.zeros(ref.shape,np.float32),np.zeros(ref.shape,np.float32)
        scored = {branch:0 for branch in extraction[case]}
        for key,item in by_case[case]:
            score_path = scores/f"{key}.npz"
            if not score_path.is_file():
                continue
            if not same_grid(nib.load(item["image"]),nib.load(scores/f"{key}.nii.gz")):
                raise ValueError(f"Scoring grid mismatch: {key}")
            with np.load(item["mapping"]) as mapping,np.load(score_path) as data:
                if tuple(mapping["native_shape"]) != ref.shape or not np.allclose(mapping["native_affine"],ref.affine):
                    raise ValueError(f"Native geometry changed: {key}")
                probs = data["probabilities"]
                if engine == "nnunet":
                    probs = probs.transpose(0,3,2,1)
                if not 0 <= plaque_channel < probs.shape[0]:
                    raise ValueError(f"Plaque probability channel missing: {key}")
                valid = mapping["valid"]
                if engine == "xyz" and "coverage" in data:
                    valid = valid & data["coverage"]
                if valid.any():
                    splat(total,weights,mapping["native_ijk"],probs[plaque_channel],valid)
                    scored[item["branch"]] += 1
        for branch,result in extraction[case].items():
            result["scored_paths"],result["success"] = scored[branch],scored[branch] > 0
        coverage = weights > 1e-6
        np.divide(total,weights,out=total,where=coverage)
        total = np.clip(total,0,1)
        probs = np.stack([np.where(coverage,1-total,0),total])
        save_prediction(out,case,ref,probs,coverage,seg=(coverage & (total >= threshold)).astype(np.uint8))
    write_json(out/"extraction.json",extraction)

"""Class overlap, surface distances, lesion matching and failure-aware summaries."""
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .data import BRANCHES, C26, branch_arrays, read_json, select, target_array, write_json
from .prediction import read_label_prediction, read_prediction, read_run


def ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def overlap(pred, gt, voxel_mm3):
    n,g = int(pred.sum()),int(gt.sum())
    intersection = int((pred & gt).sum())
    return {"gt_voxels":g,"pred_voxels":n,"dice":ratio(2*intersection,n+g) if g else None,
        "iou":ratio(intersection,n+g-intersection) if g else None,"precision":ratio(intersection,n),
        "recall":ratio(intersection,g),"absolute_volume_error_mm3":abs(n-g)*voxel_mm3,
        "relative_volume_error":ratio(abs(n-g),g),"signed_volume_error_mm3":(n-g)*voxel_mm3}


def surfaces(pred, gt, spacing, tolerance):
    from surface_distance import metrics
    if not gt.any() or not pred.any():
        return {"hd95_mm":None,"assd_mm":None,"surface_dice":0. if gt.any() else None,
                "surface_status":"empty_ground_truth" if not gt.any() else "empty_prediction"}
    distance = metrics.compute_surface_distances(gt,pred,spacing)
    a,b = metrics.compute_average_surface_distance(distance)
    ga,pa = distance["surfel_areas_gt"].sum(),distance["surfel_areas_pred"].sum()
    return {"hd95_mm":float(metrics.compute_robust_hausdorff(distance,95)),
        "assd_mm":float((a*ga+b*pa)/(ga+pa)),
        "surface_dice":float(metrics.compute_surface_dice_at_tolerance(distance,tolerance)),"surface_status":"defined"}


def lesions(pred, gt, voxel_mm3, min_iou=0.):
    truth,n_gt = ndi.label(gt,C26)
    predicted,n_pred = ndi.label(pred,C26)
    gt_sizes = np.bincount(truth.ravel(),minlength=n_gt+1)[1:]
    pred_sizes = np.bincount(predicted.ravel(),minlength=n_pred+1)[1:]
    joint = (truth > 0) & (predicted > 0)
    matched_g = np.array([],int)
    split = merge = 0
    if n_gt and n_pred:
        pairs,counts = np.unique(np.stack([truth[joint],predicted[joint]],axis=1),axis=0,return_counts=True)
        ious = np.zeros((n_gt,n_pred),np.float32)
        for (g,p),count in zip(pairs,counts):
            ious[g-1,p-1] = count/(gt_sizes[g-1]+pred_sizes[p-1]-count)
        eligible = (ious > 0) & (ious >= min_iou)
        gs,ps = linear_sum_assignment(-(eligible.astype(float)*(min(n_gt,n_pred)+1)+ious))
        matched_g = gs[eligible[gs,ps]]
        split,merge = int((eligible.sum(axis=1) > 1).sum()),int((eligible.sum(axis=0) > 1).sum())
    tp = len(matched_g)
    rows = [{"gt_component":i+1,"volume_mm3":int(size)*voxel_mm3,"detected":i in matched_g}
            for i,size in enumerate(gt_sizes)]
    return {"lesion_tp":tp,"lesion_fp":n_pred-tp,"lesion_fn":n_gt-tp,"lesion_sensitivity":ratio(tp,n_gt),
        "lesion_precision":ratio(tp,n_pred),"lesion_f1":ratio(2*tp,n_gt+n_pred) if n_gt else None,
        "lesion_split_ambiguities":split,"lesion_merge_ambiguities":merge},rows


def branch_assignment(mask, branch_masks, affine):
    coords = np.argwhere(mask)
    if not len(coords):
        return np.empty(0,int),coords
    spacing = nib.affines.voxel_sizes(affine)
    distances = []
    for branch in BRANCHES:
        support = np.argwhere(branch_masks[branch] > 0)
        if not len(support):
            raise ValueError(f"Empty evaluation branch: {branch}")
        distances.append(cKDTree(support*spacing).query(coords*spacing,workers=1)[0])
    return np.argmin(distances,axis=0),coords


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = {}
    for name in ("lumen","plaque"):
        cases = [r for r in rows if r["branch"] == "ALL" and r["class"] == name and r["scope"] == "end_to_end"]
        if not cases:
            continue
        result[name] = {}
        for key in ("dice","precision","recall","hd95_mm","assd_mm","surface_dice"):
            values = [r[key] for r in cases if r.get(key) is not None]
            result[name][key] = float(np.mean(values)) if values else None
            result[name][key+"_defined_n"] = len(values)
        positive = [r for r in cases if r["gt_voxels"]]
        successful = [r["dice"] for r in positive if r["pipeline_success"]]
        result[name].update(positive_cases=len(positive),
            successful_extraction_case_dice=float(np.mean(successful)) if successful else None,
            successful_extraction_case_n=len(successful))
    branches = [r for r in rows if r["branch"] != "ALL" and r["class"] == "plaque"]
    if branches:
        negative = [r for r in branches if not r["gt_positive_branch"]]
        success = [r for r in negative if r["pipeline_success"]]
        result["healthy_branches"] = {"n":len(negative),"successful_n":len(success),
            "specificity":ratio(sum(not r["pred_positive_branch"] for r in success),len(success)),
            "successful_negative_yield":ratio(sum(not r["pred_positive_branch"] for r in success),len(negative)),
            "abstention_rate":ratio(len(negative)-len(success),len(negative)),
            "fp_rate":ratio(sum(r["pred_positive_branch"] for r in negative),len(negative)),
            "fp_lesions_per_branch":ratio(sum(r["lesion_fp"] for r in negative),len(negative)),
            "mean_fp_volume_mm3":ratio(sum(r["healthy_branch_fp_mm3"] for r in negative),len(negative))}
        defined = [r["dice"] for r in branches if r["gt_positive_branch"] and r["pipeline_success"]]
        result["plaque"].update(successful_extraction_branch_dice=float(np.mean(defined)) if defined else None,
                                successful_extraction_branch_n=len(defined))
        result["plaque_branch_sensitivity"] = ratio(sum(r["pred_positive_branch"] and r["gt_positive_branch"] for r in branches),
                                                   sum(r["gt_positive_branch"] for r in branches))
        result["extraction_success_rate"] = ratio(sum(r["pipeline_success"] for r in branches),len(branches))
    return result


def evaluate(manifest, partition, predictions, out, *, surface_tolerance=1., lesion_iou=0.,
             threshold=None, assessment_from=None):
    predictions,out = Path(predictions),Path(out)
    metadata = read_run(predictions,manifest,partition)
    task = metadata["task"]
    if task == "branch_lumen":
        raise ValueError("Evaluate branch-lumen predictions through centerline QC or their binary lumen union.")
    if not 0 <= lesion_iou <= 1 or surface_tolerance < 0 or threshold is not None and not 0 < threshold < 1:
        raise ValueError("Invalid evaluation threshold or tolerance.")
    extraction_path = predictions/"extraction.json"
    extraction = read_json(extraction_path) if extraction_path.is_file() else None
    excluded = set(metadata.get("calibration_ids",[]))
    assessment = None
    if assessment_from:
        gate = read_json(assessment_from)
        if gate["split_sha256"] != manifest["split_sha256"] or partition != "val":
            raise ValueError("Validation assessment split mismatch.")
        excluded.update(gate["calibration_ids"])
        assessment = set(gate["assessment_ids"])
    selected = [(case,row) for case,row in select(manifest,partition)
                if case not in excluded and (assessment is None or case in assessment)]
    if not selected:
        raise ValueError("No independent assessment patients remain.")
    rows,lesion_rows = [],[]
    classes = {"multiclass":((1,"lumen"),(2,"plaque")),"lumen":((1,"lumen"),),"plaque":((1,"plaque"),)}[task]
    for case,row in selected:
        ref = nib.load(row["image"])
        spacing = nib.affines.voxel_sizes(ref.affine)
        voxel = float(np.prod(spacing))
        gt = target_array(row,task)
        missing = not (predictions/f"{case}.npz").is_file()
        if missing:
            if (predictions/f"{case}.nii.gz").is_file() or extraction is None:
                raise FileNotFoundError(f"Missing probability archive or unrecorded failure: {case}")
            pred,coverage,probability = np.zeros(ref.shape,np.uint8),np.zeros(ref.shape,bool),None
        else:
            probability,coverage = read_prediction(predictions,case,ref,task)
            pred = read_label_prediction(predictions,case,ref,task)
            if threshold is not None:
                if task != "plaque":
                    raise ValueError("Threshold override requires binary plaque probabilities.")
                pred = (coverage & (probability[1] >= threshold)).astype(np.uint8)
        if np.any(pred[~coverage]):
            raise ValueError(f"Labels extend outside sampling coverage: {case}")
        masks = branch_arrays(row,ref)
        assigned,coords = branch_assignment(pred > 0,masks,ref.affine)
        for value,name in classes:
            pmask,gmask = pred == value,gt == value
            for scope,support in (("end_to_end",np.ones(ref.shape,bool)),("covered_only",coverage)):
                p,g = pmask & support,gmask & support
                stats = {**overlap(p,g,voxel),**surfaces(p,g,spacing,surface_tolerance)}
                record = {"case":case,"branch":"ALL","class":name,"scope":scope,
                    "pipeline_success":all(r["success"] for r in extraction[case].values()) if extraction else not missing,
                    "sampling_coverage_fraction":float(coverage.mean()),**stats}
                if name == "plaque":
                    detection,details = lesions(p,g,voxel,lesion_iou)
                    record.update(detection)
                    if scope == "end_to_end":
                        lesion_rows.extend({"case":case,**item} for item in details)
                rows.append(record)
            for index,branch in enumerate(BRANCHES):
                g = masks[branch] == (1 if name == "lumen" else 2)
                branch_coords = coords[assigned == index]
                p = np.zeros(ref.shape,bool)
                p[tuple(branch_coords.T)] = pmask[tuple(branch_coords.T)]
                success = extraction[case][branch]["success"] if extraction else not missing
                record = {"case":case,"branch":branch,"class":name,"scope":"end_to_end",
                          "pipeline_success":success,**overlap(p,g,voxel)}
                if name == "plaque":
                    detection,_ = lesions(p,g,voxel,lesion_iou)
                    fp = p & ~g
                    record.update(detection,gt_positive_branch=bool(g.any()),pred_positive_branch=bool(p.any()),
                        healthy_branch_specificity=None if g.any() or not success else float(not p.any()),
                        healthy_branch_fp_mm3=None if g.any() else float(p.sum())*voxel,
                        mean_fp_probability=None if probability is None or not fp.any() else float(probability[value][fp].mean()))
                rows.append(record)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"metrics.json",rows)
    write_json(out/"lesions.json",lesion_rows)
    _csv(out/"metrics.csv",rows)
    _csv(out/"lesions.csv",lesion_rows)
    summary = summarize(rows)
    summary.update(run=metadata,cases=len(selected),evaluated_ids=[case for case,_ in selected],
        excluded_calibration_ids=sorted(excluded),surface_tolerance_mm=surface_tolerance,lesion_match_min_iou=lesion_iou,
        lesion_matching="26-connectivity, one-to-one maximum cardinality then IoU, positive overlap required",
        branch_attribution="Evaluation-only nearest annotated branch in physical space. Distant false positives remain assigned.",
        empty_gt_dice="null, healthy branches reported separately")
    write_json(out/"summary.json",summary)
    return summary

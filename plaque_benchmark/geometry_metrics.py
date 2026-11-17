"""Evaluation-only lumen distance and centerline-relative lesion coordinates."""
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .data import BRANCHES, C26, branch_arrays, select, write_json
from .evaluation import _csv, branch_assignment
from .prediction import read_label_prediction, read_run
from .smpr import transport_frames


def geometry_metrics(manifest, centerlines, predictions, out):
    partition = centerlines["partition"]
    metadata = read_run(predictions,manifest,partition)
    if metadata["task"] not in {"plaque","multiclass"} or centerlines["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Plaque task and matching geometry split required.")
    rows = []
    for case,row in select(manifest,partition):
        ref = nib.load(row["image"])
        gt = branch_arrays(row,ref)
        pred = read_label_prediction(predictions,case,ref,metadata["task"]) == (2 if metadata["task"] == "multiclass" else 1)
        assignments,coords = branch_assignment(pred,gt,ref.affine)
        for index,branch in enumerate(BRANCHES):
            result = centerlines["cases"][case][branch]
            if not result["success"]:
                rows.append({"case":case,"branch":branch,"status":"geometry_failed","reason":result.get("reason")})
                continue
            curves = []
            for path in result["paths"]:
                with np.load(path) as data:
                    points,arc,radius = data["points_ras_mm"],data["arc_mm"],data["radius_mm"]
                _,normal,binormal = transport_frames(points)
                curves.append((cKDTree(points),points,arc,radius,normal,binormal))
            lumen_tree = cKDTree(nib.affines.apply_affine(ref.affine,np.argwhere(gt[branch] == 1)))
            p = np.zeros(ref.shape,bool)
            p[tuple(coords[assignments == index].T)] = True
            for source,mask in (("prediction",p),("ground_truth",gt[branch] == 2)):
                components,count = ndi.label(mask,C26)
                for lesion_id in range(1,count+1):
                    voxels = np.argwhere(components == lesion_id)
                    xyz = nib.affines.apply_affine(ref.affine,voxels)
                    chosen = int(np.argmin([curve[0].query(xyz)[0].mean() for curve in curves]))
                    tree,points,arc,radius,normal,binormal = curves[chosen]
                    radial,nearest = tree.query(xyz)
                    delta = xyz-points[nearest]
                    theta = np.arctan2((delta*binormal[nearest]).sum(axis=1),(delta*normal[nearest]).sum(axis=1))
                    angle = np.arctan2(np.sin(theta).mean(),np.cos(theta).mean())
                    distance = lumen_tree.query(xyz)[0]
                    rows.append({"case":case,"branch":branch,"source":source,"component":lesion_id,"status":"measured",
                        "curve_index":chosen,"volume_mm3":len(voxels)*abs(float(np.linalg.det(ref.affine[:3,:3]))),
                        "longitudinal_start_mm":float(arc[nearest].min()),"longitudinal_end_mm":float(arc[nearest].max()),
                        "centerline_extent_mm":float(np.ptp(arc[nearest])),"radial_distance_mean_mm":float(radial.mean()),
                        "local_radius_mean_mm":float(radius[nearest].mean()),"cross_section_angle_mean_deg":float(np.degrees(angle)),
                        "lumen_distance_mean_mm":float(distance.mean()),"lumen_distance_max_mm":float(distance.max())})
    out = Path(out)
    write_json(out/"geometry_metrics.json",{"geometry_for_evaluation":centerlines["geometry"],
        "inference_geometry":metadata["geometry"],"cross_section_angle_reference":"Parallel-transport frame with arbitrary initial normal",
        "extent_definition":"Projection span of component voxel centers","rows":rows})
    _csv(out/"geometry_metrics.csv",rows)
    return rows

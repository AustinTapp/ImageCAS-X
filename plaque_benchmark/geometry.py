"""SHK branch adapter around the supplied VMTK extractor."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .data import BRANCHES, C26, digest, file_sha256, label_array, read_json, same_grid, select, write_json

RAS_LPS = np.array([-1,-1,1])


def resample_path(points, step_mm, radius=None):
    points = np.asarray(points,float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not np.isfinite(step_mm) or step_mm <= 0:
        raise ValueError("Expected finite Nx3 coordinates and positive sampling step.")
    increments = np.linalg.norm(np.diff(points,axis=0),axis=1)
    keep = np.r_[True,increments > 1e-6]
    arc = np.r_[0.,np.cumsum(increments[increments > 1e-6])]
    if len(arc) < 2:
        raise ValueError("Fewer than two distinct centerline points.")
    new_arc = np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/step_mm))+1))
    new = np.column_stack([np.interp(new_arc,arc,axis[keep]) for axis in points.T])
    new_radius = None if radius is None else np.interp(new_arc,arc,np.asarray(radius)[keep])
    return new,new_arc,new_radius


def skeleton_endpoints(mask, affine, return_directions=False):
    from skimage.morphology import skeletonize
    if not mask.any():
        raise ValueError("Empty vessel.")
    if ndi.label(mask,C26)[1] != 1:
        raise ValueError("Disconnected vessel. Review or use an explicit vessel-course-proxy experiment.")
    coords = np.nonzero(mask)
    lower = np.array([c.min() for c in coords])
    upper = np.array([c.max()+1 for c in coords])
    cropped = np.pad(mask[tuple(slice(a,b) for a,b in zip(lower,upper))],1)
    skeleton = skeletonize(cropped)
    ijk = np.argwhere(skeleton)
    degree = ndi.convolve(skeleton.astype(np.uint8),C26.astype(np.uint8),mode="constant")-skeleton
    ends = np.flatnonzero(degree[tuple(ijk.T)] == 1)
    if len(ends) < 2:
        raise ValueError("Fewer than two geometric endpoints.")
    points = nib.affines.apply_affine(affine,ijk+lower-1)
    distance = ndi.distance_transform_edt(cropped,sampling=nib.affines.voxel_sizes(affine))
    radius = distance[tuple(ijk.T)]
    lookup = {tuple(point):i for i,point in enumerate(ijk)}
    offsets = np.argwhere(C26)-1
    offsets = offsets[np.any(offsets != 0,axis=1)]
    estimates,directions = [],[]
    for endpoint in ends:
        current,previous,travelled,values = endpoint,-1,0.,[]
        while travelled < 6:
            next_ids = [lookup.get(tuple(ijk[current]+offset),-1) for offset in offsets]
            next_ids = [i for i in next_ids if i >= 0 and i != previous]
            if len(next_ids) != 1:
                break
            nxt = next_ids[0]
            travelled += np.linalg.norm(points[nxt]-points[current])
            if travelled >= 1:
                values.append(radius[nxt])
            previous,current = current,nxt
        estimates.append(float(np.median(values)) if values else float(radius[endpoint]))
        vector = points[endpoint]-points[current]
        if np.linalg.norm(vector) < 1e-6:
            raise ValueError("Cannot determine endpoint trajectory.")
        directions.append(vector/np.linalg.norm(vector))
    result = points[ends],np.asarray(estimates)
    return (*result,np.asarray(directions)) if return_directions else result


def order_endpoints(points, radius, seed_ras=None, min_ratio=1.20, seed_limit_mm=6.):
    if len(points) < 2:
        raise ValueError("Need two distinct endpoints.")
    if seed_ras is not None:
        seed = np.asarray(seed_ras,float)
        if seed.shape != (3,) or not np.isfinite(seed).all():
            raise ValueError("Expected three finite RAS-mm seed coordinates.")
        distances = np.linalg.norm(points-seed,axis=1)
        order = np.argsort(distances)
        if distances[order[0]] > seed_limit_mm or distances[order[1]]-distances[order[0]] < .5:
            raise ValueError("Manual seed is too far from an endpoint or assignment is ambiguous.")
        confidence = {"method":"manual_ras","distance_mm":float(distances[order[0]])}
    else:
        order = np.argsort(-radius)
        ratio = radius[order[0]]/max(radius[order[1]],1e-6)
        if ratio < min_ratio:
            raise ValueError(f"Ambiguous proximal endpoint. Caliber ratio {ratio:.2f} < {min_ratio:.2f}.")
        confidence = {"method":"caliber_heuristic","caliber_ratio":float(ratio)}
    return points[order],confidence


def _sitk_image(mask, affine):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(mask.T.astype(np.uint8))
    spacing = nib.affines.voxel_sizes(affine)
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(affine[:3,3]*RAS_LPS))
    image.SetDirection(tuple(((affine[:3,:3]/spacing)*RAS_LPS[:,None]).ravel()))
    return image


def path_qc(points, arc, radius, lumen, course, affine):
    ijk = nib.affines.apply_affine(np.linalg.inv(affine),points)
    spacing = nib.affines.voxel_sizes(affine)
    lumen_inside = ndi.map_coordinates(lumen.astype(np.uint8),ijk.T,order=0,mode="constant",cval=0).mean()
    course_inside = ndi.map_coordinates(course.astype(np.uint8),ijk.T,order=0,mode="constant",cval=0).mean()
    coords = np.nonzero(course)
    lower = np.maximum(np.array([c.min() for c in coords])-2,0)
    upper = np.minimum(np.array([c.max() for c in coords])+3,course.shape)
    cropped = course[tuple(slice(a,b) for a,b in zip(lower,upper))]
    distance = ndi.distance_transform_edt(np.pad(cropped,1),sampling=spacing)[1:-1,1:-1,1:-1]
    sampled = ndi.map_coordinates(distance,(ijk-lower).T,order=1,mode="constant",cval=0)
    crossings = sum(abs(arc[a]-arc[b]) > 3 for a,b in cKDTree(points).query_pairs(.25))
    return {"length_mm":float(arc[-1]),"lumen_inside_fraction":float(lumen_inside),
        "course_inside_fraction":float(course_inside),"boundary_distance_median_mm":float(np.median(sampled)),
        "radius_min_mm":float(radius.min()),"radius_max_mm":float(radius.max()),
        "median_radius_vs_mask_distance_mm":float(np.median(radius-sampled)),"near_self_intersections":int(crossings)}


def extract_branch(labels, affine, out, branch, modules, *, aorta=None, seed_ras=None, proxy=False,
                   step_mm=.4, min_ratio=1.20, max_aorta_gap_mm=8.):
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from . import vmtk_aorta_guided_core as core

    lumen = labels == 1
    course = labels > 0 if proxy else lumen
    if not lumen.any() or ndi.label(course,C26)[1] != 1:
        raise ValueError("Empty lumen or disconnected extraction mask.")
    config = core.parse_configuration([])
    binary = _sitk_image(course,affine)
    surface = core.binary_image_to_surface(binary,config)
    if surface.GetNumberOfPoints() < 1000:
        config = replace(config,retained_fraction=1.)
        surface = core.binary_image_to_surface(binary,config)
    history = []
    if aorta is not None and seed_ras is None:
        aorta_image = _sitk_image(aorta,affine)
        aorta_surface = core.binary_image_to_reference_surface(aorta_image)
        inlet = core.centered_aorta_coronary_interface_inlet(binary,aorta_image,surface,aorta_surface,
            interface_band_mm=config.inlet_interface_band_mm,minimum_interface_voxels=config.inlet_interface_min_voxels,
            maximum_interface_band_mm=config.inlet_interface_max_band_mm)
        if inlet.distance_mm > max_aorta_gap_mm:
            raise ValueError(f"Aorta gap exceeds ostium limit: {inlet.distance_mm:.2f} mm.")
        detection,history = core.find_stable_endpoints(Path(branch),surface,aorta_surface,aorta_image,binary,inlet,config,modules)
        endpoints = [inlet.coronary_position,*core.snap_points_to_surface(detection.endpoint_positions[1:],surface)]
        confidence = {"method":"aorta_interface","aorta_gap_mm":float(inlet.distance_mm)}
    else:
        candidates,radii,directions = skeleton_endpoints(course,affine,return_directions=True)
        points,confidence = order_endpoints(candidates,radii,seed_ras,min_ratio)
        order = cKDTree(candidates).query(points)[1]
        extension = directions[order]*(2*radii[order,None]+2*max(nib.affines.voxel_sizes(affine)))
        endpoints = core.snap_points_to_surface((points+extension)*RAS_LPS,surface)
    centerline,_,_ = core.extract_centerline(surface,endpoints,step_mm,config.simplify_voronoi,modules)
    field = centerline.GetPointData().GetArray(core.RADIUS_ARRAY_NAME)
    if field is None:
        raise ValueError("VMTK Radius field is missing.")
    all_radii = vtk_to_numpy(field)
    if not np.isfinite(all_radii).all() or (all_radii <= 0).any():
        raise ValueError("Invalid VMTK radius field.")
    qc,paths = [],[]
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        if cell.GetCellDimension() != 1:
            continue
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        points = np.array([centerline.GetPoint(j) for j in ids])*RAS_LPS
        radius = all_radii[ids]
        seed = np.asarray(endpoints[0])*RAS_LPS
        if np.linalg.norm(points[0]-seed) > np.linalg.norm(points[-1]-seed):
            points,radius = points[::-1],radius[::-1]
        jump = float(np.linalg.norm(np.diff(points,axis=0),axis=1).max(initial=0))
        points,arc,radius = resample_path(points,step_mm,radius)
        row = path_qc(points,arc,radius,lumen,course,affine)
        targets = np.asarray(endpoints[1:])*RAS_LPS
        distances = np.linalg.norm(targets-points[-1],axis=1)
        target = int(np.argmin(distances))
        span = float(np.linalg.norm(targets[target]-seed))
        row.update(max_original_jump_mm=jump,target_index=target+1,target_gap_mm=float(distances[target]),
                   source_gap_mm=float(np.linalg.norm(points[0]-seed)),expected_endpoint_span_mm=span)
        row["endpoints_reached"] = bool(row["source_gap_mm"] <= max(2.,2.5*radius[0]) and
            row["target_gap_mm"] <= max(2.,2.5*radius[-1]) and arc[-1] >= .6*span)
        row["abnormal_jump"] = jump > max(5.,3*float(radius.max()))
        row["accepted"] = row["course_inside_fraction"] >= .9 and not row["near_self_intersections"] and not row["abnormal_jump"] and row["endpoints_reached"]
        if row["accepted"]:
            if any(np.linalg.norm(points[-1]-prior[1][-1]) < .5 and abs(arc[-1]-prior[2][-1]) < 1 for prior in paths):
                row.update(accepted=False,duplicate=True)
            else:
                paths.append((i,points,arc,radius))
        qc.append(row)
    if not paths:
        raise ValueError("No VMTK path passed geometry QC.")
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    core.write_polydata_vtp(centerline,out/f"{branch}_centerline_original.vtp")
    accepted = vtk.vtkAppendPolyData()
    for cell_id,*_ in paths:
        accepted.AddInputData(core.extract_centerline_cell(centerline,cell_id))
    accepted.Update()
    core.write_polydata_vtp(accepted.GetOutput(),out/f"{branch}_centerline.vtp")
    markup = out/f"{branch}_endpoints.mrk.json"
    core.write_slicer_endpoint_markups(markup,endpoints)
    payload = read_json(markup)
    payload["markups"][0]["controlPoints"][0]["description"] = confidence["method"]
    write_json(markup,payload)
    files = []
    for i,points,arc,radius in paths:
        path = out/f"{branch}_path{i:02d}.npz"
        np.savez_compressed(path,points_ras_mm=points,arc_mm=arc,radius_mm=radius)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points*RAS_LPS,deep=True))
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(points))
        for j in range(len(points)):
            line.GetPointIds().SetId(j,j)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetLines(cells)
        for name,array in (("Radius",radius),("ArcLength",arc)):
            field = numpy_to_vtk(array,deep=True)
            field.SetName(name)
            poly.GetPointData().AddArray(field)
        core.write_polydata_vtp(poly,out/f"{branch}_path{i:02d}_resampled.vtp")
        files.append(str(path.resolve()))
    return {"success":True,"paths":files,"endpoints":len(endpoints),"orientation":confidence,"path_qc":qc,
            "length_sum_mm_shared_trunks_counted":float(sum(path[2][-1] for path in paths)),
            "dilation_history":[asdict(item) for item in history],"proxy":proxy,"surface_retained_fraction":config.retained_fraction}


def batch_centerlines(manifest, partition, out, source, *, predictions=None, aorta_dir=None,
                      aorta_label=6, seeds=None, proxy=False, step_mm=.4, min_ratio=1.20):
    from . import vmtk_aorta_guided_core as core
    from .prediction import read_run
    if source not in {"oracle","predicted"}:
        raise ValueError("Expected oracle or predicted geometry.")
    provenance = "oracle-centerline-assisted"
    if source == "predicted":
        if not predictions or seeds or proxy:
            raise ValueError("Predicted geometry requires a branch-lumen run, with automatic roots and lumen-only geometry.")
        meta = read_run(predictions,manifest,partition)
        if meta["task"] != "branch_lumen" or meta["geometry"] != "native":
            raise ValueError("Supply native LAD/LCX/RCA lumen predictions.")
        provenance = "automatic-external-aorta-unverified" if aorta_dir else "fully automatic"
    if seeds:
        provenance += "+manual-seeds"
    if proxy:
        provenance += "+vessel-course-proxy"
    seeds = read_json(seeds) if seeds else {}
    modules = core.load_vmtk_modules()
    out = Path(out).resolve()
    run = {"split_sha256":manifest["split_sha256"],"partition":partition,"geometry":provenance,"source":source,"cases":{}}
    for case,row in select(manifest,partition):
        reference = nib.load(row["image"])
        case_out = out/case
        aorta = None
        if aorta_dir:
            path = Path(aorta_dir)/f"{case}.nii.gz"
            if path.is_file():
                image = nib.load(path)
                if not same_grid(reference,image):
                    raise ValueError(f"Aorta grid mismatch: {path}")
                aorta = np.asanyarray(image.dataobj) == aorta_label
                if not aorta.any():
                    aorta = None
        pred = None
        if source == "predicted" and (Path(predictions)/f"{case}.nii.gz").is_file():
            image = nib.load(Path(predictions)/f"{case}.nii.gz")
            if not same_grid(reference,image):
                raise ValueError(f"Prediction grid mismatch: {case}")
            pred = label_array(image,(0,1,2,3))
        results = {}
        for index,branch in enumerate(BRANCHES,1):
            try:
                if source == "predicted" and pred is None:
                    raise ValueError("Missing upstream branch-lumen prediction.")
                labels = label_array(nib.load(row["branches"][branch])) if source == "oracle" else pred == index
                signature = digest({"labels":hashlib.sha256(labels.tobytes()).hexdigest(),"affine":reference.affine.tolist(),
                    "source":provenance,"seed":seeds.get(case,{}).get(branch),"step_mm":step_mm,"min_ratio":min_ratio,
                    "proxy":proxy,"aorta":None if aorta is None else hashlib.sha256(aorta.tobytes()).hexdigest(),
                    "core":file_sha256(core.__file__),"adapter":file_sha256(__file__)})
                prior_path = case_out/"centerline_qc.json"
                prior = read_json(prior_path).get(branch,{}) if prior_path.is_file() else {}
                if prior.get("signature") == signature and prior.get("success") and all(Path(path).is_file() for path in prior["paths"]):
                    results[branch] = prior
                    continue
                results[branch] = extract_branch(labels,reference.affine,case_out,branch,modules,aorta=aorta,
                    seed_ras=seeds.get(case,{}).get(branch),proxy=proxy,step_mm=step_mm,min_ratio=min_ratio)
                results[branch]["signature"] = signature
            except (ValueError,RuntimeError) as exc:
                results[branch] = {"success":False,"reason":str(exc),"paths":[]}
        write_json(case_out/"centerline_qc.json",results)
        run["cases"][case] = results
    write_json(out/"centerlines.json",run)
    return run

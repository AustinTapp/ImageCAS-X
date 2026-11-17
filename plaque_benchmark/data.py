"""SHK discovery, strict physical QA, immutable patient splits and targets."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

BRANCHES = ("LAD", "LCX", "RCA")
TASKS = {"multiclass": 3, "lumen": 2, "plaque": 2, "branch_lumen": 4}
C26 = ndi.generate_binary_structure(3, 3)
CASE_ID = re.compile(r"(?<![A-Z0-9])(SHK\d+)(?!\d)", re.I)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, lock=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if lock:
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except FileExistsError:
            if read_json(path) != value:
                raise ValueError(f"Locked metadata differs: {path}. Use a new experiment directory.")
        return
    path.write_text(payload, encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def nifti_paths(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz")))


def discover(images, masks):
    images, masks = Path(images).resolve(), Path(masks).resolve()
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Expected CT and mask directories: {images}, {masks}")
    cases = {}
    for path in nifti_paths(images):
        ids = set(CASE_ID.findall(str(path.relative_to(images)).upper()))
        if len(ids) > 1:
            raise ValueError(f"Conflicting SHK identifiers: {path}")
        if ids:
            cases.setdefault(ids.pop(), {"images": [], "masks": {}})["images"].append(str(path))
    for folder in sorted(masks.iterdir()):
        if not folder.is_dir() or not CASE_ID.fullmatch(folder.name):
            continue
        case = cases.setdefault(folder.name.upper(), {"images": [], "masks": {}})
        for branch in BRANCHES:
            found = [str(p) for p in nifti_paths(folder) if p.name.upper() in {f"{branch}.NII", f"{branch}.NII.GZ"}]
            if found:
                case["masks"][branch] = found
    if not cases:
        raise ValueError("No SHK patients discovered.")
    return dict(sorted(cases.items()))


def geometry(img):
    if img.ndim != 3 or not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-9:
        raise ValueError("Expected a scalar 3D image with invertible physical geometry.")
    spacing = nib.affines.voxel_sizes(img.affine)
    direction = img.affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Sheared geometry requires explicit review.")
    qform, qcode = img.get_qform(coded=True)
    sform, scode = img.get_sform(coded=True)
    if not (qcode or scode):
        raise ValueError("Neither qform nor sform defines physical geometry.")
    if qcode and scode and not np.allclose(qform, sform, atol=.005, rtol=0):
        raise ValueError("qform and sform disagree.")
    units = img.header.get_xyzt_units()[0]
    if units not in {"mm", "unknown"}:
        raise ValueError(f"Convert spatial units explicitly to millimeters: {units}")
    return {"shape_xyz": list(img.shape), "affine_ras_mm": img.affine.tolist(),
            "spacing_xyz_mm": spacing.tolist(), "origin_ras_mm": img.affine[:3, 3].tolist(),
            "direction_ras": direction.tolist(), "orientation": list(nib.aff2axcodes(img.affine)),
            "qform_code": int(qcode), "sform_code": int(scode), "declared_units": units}


def same_grid(reference, other):
    geometry(reference)
    geometry(other)
    return reference.shape == other.shape and np.allclose(reference.affine, other.affine, atol=.005, rtol=0)


def label_array(img, allowed=(0, 1, 2)):
    array = np.asanyarray(img.dataobj)
    if not np.isin(array, allowed).all():
        raise ValueError(f"Unexpected labels. Allowed values are {allowed}.")
    return array.astype(np.uint8, copy=False)


def save_image(path, array, affine):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


def _counts(mask):
    components, count = ndi.label(mask, C26)
    sizes = np.bincount(components.ravel())[1:]
    return int(count), int(sizes.max(initial=0))


def audit(images, masks, review=None):
    reviewed = {}
    if review:
        with Path(review).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                case = row["case"].upper()
                if row["status"] not in {"accept", "exclude"} or case in reviewed:
                    raise ValueError(f"Invalid or duplicate review decision: {case}")
                reviewed[case] = row["status"]
    cases = []
    for case, sources in discover(images, masks).items():
        errors, flags, branches = [], [], {}
        row = {"case": case, "image": None, "branches": branches, "errors": errors, "flags": flags,
               "review": reviewed.get(case, "pending"), "sources": sources}
        image = None
        if len(sources["images"]) != 1:
            errors.append("missing_ct" if not sources["images"] else "ambiguous_ct")
        else:
            row["image"] = sources["images"][0]
            try:
                image = nib.load(row["image"])
                row["geometry"] = geometry(image)
                row["image_sha256"] = file_sha256(row["image"])
                ct = image.get_fdata(dtype=np.float32)
                if not np.isfinite(ct).all():
                    raise ValueError("Nonfinite CT values.")
            except (ValueError, OSError, nib.filebasedimages.ImageFileError) as exc:
                image = None
                errors.append(f"invalid_ct: {exc}")
        arrays = {}
        for branch in BRANCHES:
            paths = sources["masks"].get(branch, [])
            if len(paths) != 1:
                errors.append(f"{branch}: {'missing' if not paths else 'ambiguous'} annotation")
                continue
            try:
                mask = nib.load(paths[0])
                info = geometry(mask)
                labels = label_array(mask)
                lumen, plaque = int((labels == 1).sum()), int((labels == 2).sum())
                branches[branch] = {"path": paths[0], "sha256": file_sha256(paths[0]), "geometry": info,
                                    "lumen_voxels": lumen, "plaque_voxels": plaque, "plaque_free": bool(lumen and not plaque)}
                if image is None or not same_grid(image, mask):
                    errors.append(f"{branch}: CT/mask grid mismatch")
                    continue
                arrays[branch] = labels
                if not lumen:
                    errors.append(f"{branch}: {'empty' if not plaque else 'no lumen'} annotation")
                    continue
                lum_n, lum_max = _counts(labels == 1)
                course_n, course_max = _counts(labels > 0)
                plaque_n, _ = _counts(labels == 2)
                voxel = float(np.prod(nib.affines.voxel_sizes(mask.affine)))
                branches[branch].update(lumen_components=lum_n, lumen_largest_fraction=lum_max/lumen,
                    course_components=course_n, course_largest_fraction=course_max/(lumen+plaque),
                    plaque_components=plaque_n, lumen_volume_mm3=lumen*voxel, plaque_volume_mm3=plaque*voxel,
                    lumen_hu_percentiles=np.percentile(ct[labels == 1], [1, 50, 99]).tolist())
                if plaque:
                    branches[branch]["plaque_hu_percentiles"] = np.percentile(ct[labels == 2], [1, 50, 99]).tolist()
                if lum_n > 1 or course_n > 1:
                    flags.append(f"{branch}: disconnected lumen or vessel course")
                coords = np.nonzero(labels)
                if any(axis.min() == 0 or axis.max() == labels.shape[i]-1 for i, axis in enumerate(coords)):
                    flags.append(f"{branch}: annotation touches CT boundary")
            except (ValueError, OSError, nib.filebasedimages.ImageFileError) as exc:
                errors.append(f"{branch}: {exc}")
        row["overlap"] = {}
        for i, a in enumerate(arrays):
            for b in list(arrays)[i+1:]:
                overlap = (arrays[a] > 0) & (arrays[b] > 0)
                if overlap.any():
                    conflicts = int((arrays[a][overlap] != arrays[b][overlap]).sum())
                    row["overlap"][f"{a}/{b}"] = {"voxels": int(overlap.sum()), "conflicting_labels": conflicts}
                    flags.append(f"{a}/{b}: overlapping branches")
                    if conflicts:
                        errors.append(f"{a}/{b}: conflicting lumen/plaque labels")
        row["status"] = "excluded" if errors or row["review"] == "exclude" else (
            "review" if flags and row["review"] != "accept" else "eligible")
        cases.append(row)
    return {"schema": 1, "cases": cases,
            "physical_review": "Header agreement cannot establish anatomical alignment or complete annotation coverage."}


def make_split(report, seed=2028, fractions=(.70, .15, .15), require_reviewed=False):
    rows = [row for row in report["cases"] if row["status"] == "eligible" and
            (not require_reviewed or row["review"] == "accept")]
    n = len(rows)
    fractions = np.asarray(fractions, float)
    if n < 3 or fractions.shape != (3,) or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("Need three eligible patients and three positive fractions summing to one.")
    sizes = np.maximum(1, np.floor(n*fractions).astype(int))
    while sizes.sum() < n:
        sizes[len(sizes)-1-np.argmax((n*fractions-sizes)[::-1])] += 1
    while sizes.sum() > n:
        sizes[np.argmax(np.where(sizes > 1, sizes-n*fractions, -np.inf))] -= 1
    patterns = np.array([[bool(row["branches"][b]["plaque_voxels"]) for b in BRANCHES] for row in rows])
    features = np.column_stack([patterns.any(axis=1), ~patterns.any(axis=1), patterns]).astype(int)
    expected = sizes[:, None] * features.sum(axis=0) / n
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(1024):
        groups = np.split(rng.permutation(n), np.cumsum(sizes)[:-1])
        observed = np.stack([features[group].sum(axis=0) for group in groups])
        score = np.sum((observed-expected)**2/np.maximum(1, expected))
        if score < best_score:
            best, best_score = groups, score
    return {"schema": 1, "audit_sha256": digest(report), "seed": seed, "require_reviewed": require_reviewed,
            "partitions": {name: sorted(rows[i]["case"] for i in group)
                           for name, group in zip(("train", "val", "test"), best)}}


def prepare(report, split, out):
    if split["audit_sha256"] != digest(report):
        raise ValueError("Audit and locked split differ.")
    out = Path(out).resolve()
    manifest = {"schema": 1, "split_sha256": digest(split), "cases": {}}
    inventory = {row["case"]: row for row in report["cases"]}
    for partition, ids in split["partitions"].items():
        for case in ids:
            row = inventory[case]
            if file_sha256(row["image"]) != row["image_sha256"]:
                raise ValueError(f"CT changed after audit: {case}")
            ref = nib.load(row["image"])
            branches = {b: row["branches"][b]["path"] for b in BRANCHES}
            combined, branch_lumen = np.zeros(ref.shape, np.uint8), np.zeros(ref.shape, np.uint8)
            for index, (branch, path) in enumerate(branches.items(), 1):
                if file_sha256(path) != row["branches"][branch]["sha256"]:
                    raise ValueError(f"Mask changed after audit: {path}")
                image = nib.load(path)
                if not same_grid(ref, image):
                    raise ValueError(f"Changed source geometry: {path}")
                labels = label_array(image)
                if np.any((combined > 0) & (labels > 0) & (combined != labels)) or not (labels == 1).any():
                    raise ValueError(f"Conflicting labels or missing lumen: {path}")
                np.maximum(combined, labels, out=combined)
                lumen = labels == 1
                branch_lumen[lumen] = np.where(branch_lumen[lumen] == 0, index, 255)
            label_path, branch_path = out/"labels"/f"{case}.nii.gz", out/"branch_lumen"/f"{case}.nii.gz"
            save_image(label_path, combined, ref.affine)
            save_image(branch_path, branch_lumen, ref.affine)
            manifest["cases"][case] = {"partition": partition, "image": row["image"], "branches": branches,
                "label": str(label_path), "branch_lumen": str(branch_path), "image_sha256": row["image_sha256"],
                "annotation_status": row["review"]}
    write_json(out/"dataset.json", manifest, lock=True)
    return manifest


def select(manifest, partition):
    return [(case, row) for case, row in sorted(manifest["cases"].items()) if row["partition"] == partition]


def target_array(row, task):
    path = row["branch_lumen"] if task == "branch_lumen" else row["label"]
    target, reference = nib.load(path), nib.load(row["image"])
    if not same_grid(reference, target):
        raise ValueError(f"CT/target grid mismatch: {path}")
    if task == "branch_lumen":
        return label_array(target, (0, 1, 2, 3, 255))
    labels = label_array(target)
    if task == "multiclass":
        return labels
    return (labels == {"lumen": 1, "plaque": 2}[task]).astype(np.uint8)


def branch_arrays(row, reference):
    result = {}
    for branch, path in row["branches"].items():
        image = nib.load(path)
        if not same_grid(reference, image):
            raise ValueError(f"Evaluation branch grid mismatch: {path}")
        result[branch] = label_array(image)
    return result


def training_signature(manifest):
    return digest({"split_sha256": manifest["split_sha256"], "geometry": manifest.get("geometry", "native"),
                   "cases": {case: row for case, row in manifest["cases"].items() if row["partition"] in {"train", "val"}}})

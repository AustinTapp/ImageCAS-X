"""File, geometry and review operations; independent of the GUI and PyTorch."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np

DEFAULT_RUNS = ['PlaquePilot_away80_tuned_ctta25', 'PlaquePilot_nnfix80_ctta25']
CATEGORIES = ('approved', 'rejected', 'needs_review', 'unreviewed')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def resolve_source(path, manifest_path):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd()/path, Path(manifest_path).parent/path]
    return next((p.resolve() for p in candidates if p.is_file()), candidates[0].resolve())


@dataclass
class PredictionSet:
    name: str
    root: Path
    task: str
    metadata: dict
    metrics: dict


def load_metrics(report):
    path = Path(report)/'metrics.csv'
    if not path.is_file():
        return {}
    result = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row.get('branch') == 'ALL' and row.get('scope') == 'end_to_end':
                result.setdefault(row['case'], {})[row['class']] = row
    return result


def discover_predictions(bench, manifest, partition='val', runs=None, predictions=None):
    """Never guess which checkpoint to use when several fingerprints exist."""
    bench = Path(bench)
    found = []
    explicit = list(predictions or [])
    if explicit:
        entries = []
        for entry in explicit:
            name, separator, root = entry.partition('=')
            if not separator or not name or not root:
                raise ValueError('--prediction must be NAME=DIRECTORY')
            entries.append((name, Path(root), None))
    else:
        entries = []
        selected = runs or [r for r in DEFAULT_RUNS if (bench/'runner'/r).is_dir()]
        if not selected:
            raise FileNotFoundError('No default runs found. Use --runs RUN_NAME ... or --prediction NAME=DIRECTORY.')
        for name in selected:
            run = bench/'runner'/name
            if not (run/'predictions').is_dir():
                raise FileNotFoundError(f'No predictions directory: {run}')
            for model in sorted((run/'predictions').iterdir()):
                if not model.is_dir():
                    continue
                candidates = [p.parent for p in model.glob('*/run.json')]
                if (model/'run.json').is_file():
                    candidates.append(model)
                if len(candidates) != 1:
                    raise ValueError(f'{model}: found {len(candidates)} prediction fingerprints. Select one with --prediction NAME=DIRECTORY.')
                entries.append((f'{name}/{model.name}', candidates[0], run/'reports'/model.name))
    for name, root, report in entries:
        meta = read_json(root/'run.json')
        if meta.get('split_sha256') != manifest['split_sha256'] or meta.get('partition') != partition:
            raise ValueError(f'Split/partition mismatch for {name}: {root}')
        task = meta.get('task')
        if task not in ('multiclass', 'plaque', 'lumen', 'branch_lumen'):
            raise ValueError(f'Unsupported prediction task {task!r}')
        if report is None:
            # Standard runner layout: predictions/MODEL/FINGERPRINT -> reports/MODEL.
            if root.parent.parent.name == 'predictions':
                report = root.parent.parent.parent/'reports'/root.parent.name
        found.append(PredictionSet(name, root.resolve(), task, meta, load_metrics(report) if report else {}))
    if not found or len({p.name for p in found}) != len(found):
        raise ValueError('No prediction sets, or duplicate display names.')
    return found


def source_paths(manifest_path, row, predictions, case):
    return dict(ct=resolve_source(row['image'], manifest_path),
                ground_truth=resolve_source(row['label'], manifest_path),
                prediction=predictions.root/f'{case}.nii.gz')


def checked_image(path):
    image = nib.load(str(path))
    if image.ndim != 3 or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-9:
        raise ValueError(f'Expected scalar 3D NIfTI with invertible physical geometry: {path}')
    if image.header.get_xyzt_units()[0] not in ('unknown', 'mm'):
        raise ValueError(f'Convert spatial units to mm before review: {path}')
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if qc and sc and not np.allclose(q, s, atol=.005, rtol=0):
        raise ValueError(f'qform/sform conflict: {path}. Resolve which geometry is correct before viewing.')
    return image


def same_grid(a, b):
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=.005, rtol=0)


def checked_labels(image, allowed):
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.isin(data, allowed).all():
        raise ValueError(f'Unexpected label values: expected {allowed}; got {np.unique(data)[:20]}')
    return data.astype(np.uint8)


@dataclass
class SharedVolume:
    case: str
    original: object
    reference: object
    ct: np.ndarray
    gt: np.ndarray
    warnings: list
    resample_labels: bool


@dataclass
class ReviewVolume:
    shared: SharedVolume
    pred: np.ndarray
    metrics: dict
    warnings: list


def load_volume(case, paths, task='multiclass', resample_labels=False, shared=None):
    warnings = []
    if shared is None or shared.case != case or shared.resample_labels != resample_labels:
        original = checked_image(paths['ct'])
        reference = nib.as_closest_canonical(original)
        # True orthogonal anatomical planes, including oblique acquisitions.
        if not np.allclose(reference.affine[:3, :3], np.diag(nib.affines.voxel_sizes(reference.affine)), atol=1e-4):
            reference = resample_to_output(original, voxel_sizes=nib.affines.voxel_sizes(reference.affine), order=1, cval=-1024)
            warnings.append('Oblique CT resampled onto an orthogonal RAS grid for DISPLAY only.')
        ct = reference.get_fdata(dtype=np.float32)
        if not np.isfinite(ct).all():
            raise ValueError('CT contains nonfinite values.')
        gt_image = checked_image(paths['ground_truth'])
        gt_values = checked_labels(gt_image, (0, 1, 2, 255))
        if not same_grid(gt_image, original):
            if not resample_labels:
                raise ValueError('Ground truth grid differs from CT. Enable display resampling only if its physical affine is trustworthy.')
            warnings.append('Ground truth grid differs; nearest-neighbor physical resampling for display.')
        if not same_grid(gt_image, reference):
            gt_values = np.asanyarray(resample_from_to(nib.Nifti1Image(gt_values, gt_image.affine), reference, order=0).dataobj).astype(np.uint8)
        shared = SharedVolume(case, original, reference, ct, gt_values, warnings.copy(), resample_labels)
    warnings = list(shared.warnings)
    pred_image = checked_image(paths['prediction'])
    allowed = (0, 1, 2, 255) if task == 'multiclass' else (0, 1, 2, 3, 255) if task == 'branch_lumen' else (0, 1, 255)
    pred = checked_labels(pred_image, allowed)
    if not same_grid(pred_image, shared.original):
        if not resample_labels:
            raise ValueError('Prediction grid differs from CT. Enable display resampling only if its physical affine is trustworthy.')
        warnings.append('Prediction grid differs; nearest-neighbor physical resampling for display.')
    if not same_grid(pred_image, shared.reference):
        pred = np.asanyarray(resample_from_to(nib.Nifti1Image(pred, pred_image.affine), shared.reference, order=0).dataobj).astype(np.uint8)
    ignored = pred == 255
    if task == 'plaque':
        pred = (pred == 1).astype(np.uint8)*2
    elif task == 'branch_lumen':
        pred = ((pred > 0) & ~ignored).astype(np.uint8)
    pred[ignored] = 255
    stats = {}
    for value, name in ((1, 'lumen'), (2, 'plaque')):
        if task == 'plaque' and value == 1 or task in ('lumen', 'branch_lumen') and value == 2:
            continue
        p, g = pred == value, shared.gt == value
        n, m, intersection = int(p.sum()), int(g.sum()), int((p & g).sum())
        stats[name] = dict(dice=2*intersection/(n+m) if m else None,
                           precision=intersection/n if n else None, recall=intersection/m if m else None,
                           gt_voxels=m, pred_voxels=n)
    return ReviewVolume(shared, pred, stats, warnings)


# Display arrays are row-major, in radiological convention.
PLANES = {'Axial': (2, 0, 1, True), 'Coronal': (1, 0, 2, True), 'Sagittal': (0, 1, 2, False)}


def slice_array(array, indices, plane):
    axis, horizontal, vertical, flip_x = PLANES[plane]
    result = np.take(array, int(indices[axis]), axis=axis).T[::-1, :]
    return result[:, ::-1] if flip_x else result


def screen_to_indices(x, y, shape, indices, plane):
    axis, horizontal, vertical, flip_x = PLANES[plane]
    result = list(indices)
    result[horizontal] = int(np.clip(round(shape[horizontal]-1-x if flip_x else x), 0, shape[horizontal]-1))
    result[vertical] = int(np.clip(round(shape[vertical]-1-y), 0, shape[vertical]-1))
    return result


def safe_name(value):
    clean = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')[:90] or 'item'
    return clean + '_' + hashlib.sha256(value.encode()).hexdigest()[:8]


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class ReviewStore:
    """Categories contain copied files; originals are never moved or hard-linked.

    CT/GT/prediction are copied once into a private asset cache, then hard-linked
    from that cache into category packages (copy fallback). Reclassification
    replaces only the review package. A per-store lock prevents concurrent edits.
    """
    def __init__(self, root, references_only=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.references_only = references_only

    def records(self):
        path = self.root/'reviews.json'
        return read_json(path) if path.exists() else {}

    @staticmethod
    def key(model, case):
        return f'{model}::{case}'

    def _asset(self, source):
        source = Path(source).resolve()
        stat = source.stat()
        identity = [str(source), stat.st_size, stat.st_mtime_ns]
        token = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        destination = self.root/'assets'/token/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temp = destination.with_name(destination.name + '.copying')
            try:
                shutil.copy2(source, temp)
                after = source.stat()
                if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f'Source changed during export: {source}')
                os.replace(temp, destination)
            finally:
                temp.unlink(missing_ok=True)
        return destination

    def decide(self, model, case, category, paths, *, notes='', reason='', view=None, reviewer=''):
        if category not in CATEGORIES:
            raise ValueError(f'Unknown category: {category}')
        with exclusive(self.root/'.review.lock'):
            records = self.records()
            key = self.key(model, case)
            previous = records.get(key)
            folder = Path(category)/safe_name(model)/safe_name(case)
            target = self.root/folder
            old = self.root/previous['folder'] if previous else None
            record = dict(model=model, case=case, category=category, notes=notes, reason=reason,
                          reviewer=reviewer, utc=datetime.now(timezone.utc).isoformat(),
                          sources={k:str(Path(v).resolve()) for k,v in paths.items()},
                          missing=[k for k,v in paths.items() if not Path(v).is_file()],
                          folder=folder.as_posix(), view=view or {}, references_only=self.references_only)
            # Stage complete package before changing either category or index.
            staging_root = self.root/'.staging'; staging_root.mkdir(exist_ok=True)
            stage = Path(tempfile.mkdtemp(dir=staging_root))
            backup = None
            installed = committed = False
            try:
                if not self.references_only:
                    for label, path in paths.items():
                        if not Path(path).is_file():
                            continue
                        suffix = '.nii.gz' if str(path).lower().endswith('.nii.gz') else Path(path).suffix
                        asset = self._asset(path)
                        dest = stage/(label+suffix)
                        try:
                            os.link(asset, dest)
                        except OSError:
                            shutil.copy2(asset, dest)
                atomic_json(stage/'review.json', record)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if old != target:
                        raise FileExistsError(f'Unindexed existing review package: {target}')
                    backup = stage.with_name(stage.name + '.backup')
                    os.replace(target, backup)
                os.replace(stage, target)
                installed = True
                records[key] = record
                try:
                    atomic_json(self.root/'reviews.json', records)
                except BaseException:
                    shutil.rmtree(target)
                    installed = False
                    if backup: os.replace(backup, target)
                    backup = None
                    raise
                committed = True
                warnings = []
                for obsolete in [old if old != target else None, backup]:
                    if obsolete and obsolete.exists():
                        try: shutil.rmtree(obsolete)
                        except OSError as exc: warnings.append(f'Review saved; could not remove old review package {obsolete}: {exc}')
                try:
                    with (self.root/'history.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(record) + '\n')
                    fields = ['model', 'case', 'category', 'reason', 'notes', 'reviewer', 'utc', 'folder']
                    temp = self.root/'reviews.csv.tmp'
                    with temp.open('w', newline='', encoding='utf-8') as stream:
                        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
                        writer.writeheader(); writer.writerows(records.values())
                    os.replace(temp, self.root/'reviews.csv')
                except OSError as exc:
                    # Excel can lock the CSV on Windows. reviews.json remains authoritative.
                    warnings.append(f'Review saved; history/CSV export warning: {exc}')
                record['export_warnings'] = warnings
                return record
            finally:
                if not committed:
                    if installed and target.exists(): shutil.rmtree(target)
                    if backup and backup.exists(): os.replace(backup, target)
                if stage.exists(): shutil.rmtree(stage)


def make_demo(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    shape = (72, 80, 56)
    x, y, z = np.indices(shape)
    ct = (-900 + 950*np.exp(-((x-36)**2/600 + (y-40)**2/850 + (z-28)**2/600))).astype(np.float32)
    gt = np.zeros(shape, np.uint8)
    vessel = (x-35-5*np.sin(z/12))**2 + (y-40)**2 < 30
    gt[vessel] = 1
    gt[vessel & (z > 24) & (z < 34) & (x > 36)] = 2
    ct[gt == 1] = 380; ct[gt == 2] = 160
    affine = np.diag([.7, .7, 1.2, 1])
    cases = {}
    for index in (1, 2):
        case = f'DEMO{index:03d}'
        image, label = root/f'{case}_ct.nii.gz', root/f'{case}_gt.nii.gz'
        for path, array in ((image, ct), (label, gt)):
            nii = nib.Nifti1Image(array, affine); nii.header.set_xyzt_units('mm'); nib.save(nii, path)
        cases[case] = dict(image=str(image), label=str(label), partition='val')
        for name, offset in (('demo_good', 0), ('demo_shifted', 3)):
            pred_root = root/name; pred_root.mkdir(exist_ok=True)
            pred = np.roll(gt, offset, axis=0)
            nii = nib.Nifti1Image(pred, affine); nii.header.set_xyzt_units('mm'); nib.save(nii, pred_root/f'{case}.nii.gz')
            atomic_json(pred_root/'run.json', dict(split_sha256='synthetic-demo', task='multiclass', partition='val'))
    atomic_json(root/'dataset.json', dict(split_sha256='synthetic-demo', cases=cases))
    return root/'dataset.json', [f'{name}={root/name}' for name in ('demo_good', 'demo_shifted')]

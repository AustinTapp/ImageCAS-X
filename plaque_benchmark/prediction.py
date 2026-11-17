"""Native-grid probability interchange and provenance."""
from pathlib import Path
import nibabel as nib
import numpy as np
from .data import TASKS, label_array, read_json, same_grid, save_image, write_json


def run_metadata(out, manifest, partition, task, geometry="native", **extra):
    record = {"schema": 1, "split_sha256": manifest["split_sha256"], "partition": partition,
              "task": task, "geometry": geometry, **extra}
    write_json(Path(out)/"run.json", record, lock=True)
    return record


def read_run(root, manifest, partition):
    record = read_json(Path(root)/"run.json")
    if record["split_sha256"] != manifest["split_sha256"] or record["partition"] != partition:
        raise ValueError(f"Prediction split/partition mismatch: {root}")
    return record


def read_prediction(root, case, reference, task):
    with np.load(Path(root)/f"{case}.npz") as data:
        probs = data["probabilities"].astype(np.float32)
        coverage, affine = data["coverage"].astype(bool), data["affine"]
    if probs.shape != (TASKS[task], *reference.shape) or coverage.shape != reference.shape:
        raise ValueError(f"Prediction dimensions mismatch: {case}")
    if not np.allclose(reference.affine, affine, atol=.005, rtol=0):
        raise ValueError(f"Prediction affine mismatch: {case}")
    if not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
        raise ValueError(f"Invalid probabilities: {case}")
    if not np.allclose(probs.sum(axis=0), coverage, atol=.003):
        raise ValueError(f"Probabilities must sum to one on coverage and zero elsewhere: {case}")
    return probs, coverage


def save_prediction(root, case, reference, probs, covered=None, *, seg=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if covered is None:
        covered = np.ones(reference.shape, bool)
    if seg is None:
        seg = probs.argmax(axis=0).astype(np.uint8)
    save_image(root/f"{case}.nii.gz", seg, reference.affine)
    np.savez_compressed(root/f"{case}.npz", probabilities=probs.astype(np.float16),
                        coverage=covered, affine=reference.affine)


def read_label_prediction(root, case, reference, task):
    image = nib.load(Path(root)/f"{case}.nii.gz")
    if not same_grid(reference, image):
        raise ValueError(f"Prediction label grid mismatch: {case}")
    return label_array(image, tuple(range(TASKS[task])))

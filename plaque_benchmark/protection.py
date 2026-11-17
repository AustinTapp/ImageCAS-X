"""Freeze model identities and evaluation settings before test access."""
from pathlib import Path
from .data import file_sha256, read_json, write_json


def freeze(manifest, out, *, checkpoints=(), runs=(), decision="", surface_tolerance_mm=1., lesion_iou=0.):
    if not decision.strip():
        raise ValueError("Record the model-selection decision.")
    candidates = [file_sha256(path) for path in checkpoints]
    for root in runs:
        candidate = read_json(Path(root)/"run.json").get("candidate_id")
        if not candidate:
            raise ValueError(f"Missing candidate identity: {root}")
        candidates.append(candidate)
    if not candidates:
        raise ValueError("Supply at least one checkpoint or validation run.")
    value = {"split_sha256": manifest["split_sha256"], "candidate_ids": sorted(set(candidates)),
             "decision": decision, "evaluation_protocol": {"surface_tolerance_mm": surface_tolerance_mm, "lesion_iou": lesion_iou}}
    write_json(out, value, lock=True)
    return value


def guard(manifest, partition, frozen, candidate=None):
    if partition != "test":
        return
    if not frozen:
        raise ValueError("Test access requires --frozen and a concrete promotion record.")
    value = read_json(frozen)
    if value["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Promotion split mismatch.")
    if candidate is not None and candidate not in value["candidate_ids"]:
        raise ValueError("Candidate was not promoted before test access.")

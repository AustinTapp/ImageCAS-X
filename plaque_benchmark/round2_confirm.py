"""Frozen PlaquePilot confirmation and prediction-only specialist fusion."""
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

from .data import C26, digest, file_sha256, read_json, select, write_json
from .prediction import read_label_prediction, read_prediction, read_run, run_metadata, save_prediction

SEED = 2028
PROTOCOL = {"surface_tolerance": 1.0, "lesion_iou": 0.0}
MODELS = ("nnunet", "nnunet_resenc", "ffr_unet")
# All distances/volumes are physical. The 3 mm rule is exploratory until confirmed.
RULES = {
    "nnunet_ffr_050": {"plaque": ["nnunet", "ffr_unet"]},
    "nnunet_ffr_050_min1": {"plaque": ["nnunet", "ffr_unet"], "min_mm3": 1.0},
    "nnunet_resenc_050": {"plaque": ["nnunet", "nnunet_resenc"]},
    "resenc_lumen_ffr_050": {"lumen": "nnunet_resenc", "plaque": ["ffr_unet"]},
    "resenc_lumen_ffr_050_gate3mm": {"lumen": "nnunet_resenc", "plaque": ["ffr_unet"], "distance_mm": 3.0},
    "resenc_lumen_nnunet_ffr_050": {"lumen": "nnunet_resenc", "plaque": ["nnunet", "ffr_unet"]},
}


def checkpoint_choices(bench, overrides, results_root):
    runner = Path(bench) / "runner"
    result = {k: Path(v).resolve() for k, v in overrides.items() if v}
    run = runner / "PlaquePilot_nnfix80_ctta25"
    art = runner / "PlaquePilot_away80_tuned_ctta25"
    if "ffr_unet" not in result:
        result["ffr_unet"] = art / "training/ffr_unet/model/best.pt"
    for model in ("nnunet", "nnunet_resenc"):
        if model in result:
            continue
        training = read_json(run / "training" / model / "specification.json")
        signature = training["signature"]
        nn_paths = training["context"].get("environment", {}).get("paths", {})
        root = Path(results_root or nn_paths.get("nnUNet_results") or "")
        if not results_root and not nn_paths.get("nnUNet_results"):
            raise ValueError("Supply --nnunet-results or explicit --nnunet and --nnunet-resenc checkpoint paths")
        pattern = f"Dataset*/*Screen{signature[:18]}__*__3d_fullres/fold_0/checkpoint_best."
        safe = list(root.glob(pattern + "safetensors"))
        found = safe or list(root.glob(pattern + "pth"))
        if len(found) != 1:
            raise ValueError(f"Expected one original {model} checkpoint under {root}. Supply --{model.replace('_','-')}.")
        result[model] = found[0]
    for model, location in result.items():
        if not location.is_file():
            raise FileNotFoundError(location)
        directory = art if model == "ffr_unet" else run
        summary = directory / "reports" / model / "summary.json"
        if summary.is_file():
            test_id = read_json(summary)["run"]["checkpoint_sha256"]
            if file_sha256(location) != test_id:
                raise ValueError(f"{model} checkpoint differs from the validation report")
    return result


def sources_for(manifest, partition, archive, checkpoints, bench):
    sources = {}
    for name, checkpoint in checkpoints.items():
        sha = file_sha256(checkpoint) if checkpoint else None
        root = archive / "source_predictions" / name
        if partition == "val" and not root.exists():
            run = "PlaquePilot_away80_tuned_ctta25" if name == "ffr_unet" else "PlaquePilot_nnfix80_ctta25"
            candidates = list((bench / "runner" / run / "predictions" / name).glob("*/run.json"))
            matches = [p.parent for p in candidates if read_json(p).get("checkpoint_sha256") == sha]
            if len(matches) == 1:
                root = matches[0]
        sources[name] = {"checkpoint": str(checkpoint), "sha256": sha, "predictions": str(root)}
        if name != "ffr_unet":
            model_folder = checkpoint.parent.parent
            for filename in ("plans.json", "dataset.json"):
                sources[name][filename] = file_sha256(model_folder / filename)
            if read_json(model_folder / "plans.json")["image_reader_writer"] != "SimpleITKIO":
                raise ValueError("Confirmation importer expects the existing SimpleITKIO nnU-Net plans")
    return sources


def confirmation_record(dataset_path, manifest, sources, decision):
    if not decision.strip():
        raise ValueError("Specify the prespecified primary comparison in --decision")
    rules = {name: {"threshold": .5, **rule} for name, rule in RULES.items()}
    code_files = ("round2_confirm.py", "native.py", "evaluation.py", "prediction.py",
                  "external.py", "data.py", "../inference.py")
    identities = {key: digest({"rule": rule, "sources": {m: sources[m]["sha256"]
                               for m in set(rule["plaque"]) | ({rule["lumen"]} if "lumen" in rule else set())}})
                  for key, rule in rules.items()}
    return {
        "split_sha256": manifest["split_sha256"], "dataset_file_sha256": file_sha256(dataset_path),
        "candidate_ids": sorted([*identities.values(), *[v["sha256"] for v in sources.values()]]),
        "evaluation_protocol": {"surface_tolerance_mm": PROTOCOL["surface_tolerance"], "lesion_iou": 0.0},
        "decision": decision, "sources": sources, "rules": rules, "identities": identities,
        "implementation": {n: file_sha256(Path(__file__).parent / n) for n in code_files},
    }


def complete_source(path, manifest, partition, sha):
    if not (path / "run.json").is_file():
        return False
    if read_run(path, manifest, partition).get("checkpoint_sha256") != sha:
        raise ValueError(f"Checkpoint mismatch at {path}")
    return all((path / f"{case}.npz").is_file() and (path / f"{case}.nii.gz").is_file()
               for case, _ in select(manifest, partition))


def obtain_sources(manifest, partition, archive, sources, device, nnunet_python):
    from .external import export_inputs, import_nnunet

    paths = {}
    for name, source in sources.items():
        path = Path(source["predictions"])
        checkpoint = Path(source["checkpoint"])
        if not complete_source(path, manifest, partition, source["sha256"]):
            if name == "ffr_unet":
                from .native import infer
                infer(manifest, partition, checkpoint, path, device=device)
            else:
                inputs = archive / "nnunet_inputs"
                export_inputs(manifest, partition, inputs)
                raw = archive / "nnunet_raw_predictions" / name
                raw.mkdir(parents=True, exist_ok=True)
                fold = checkpoint.parent.name.removeprefix("fold_")
                import_call = "from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder; predict_entry_point_modelfolder()"
                subprocess.run([nnunet_python or sys.executable, "-c", import_call, "-i", str(inputs), "-o", str(raw),
                                "-m", str(checkpoint.parent.parent), "-f", fold, "-chk", checkpoint.name,
                                "-device", device, "-npp", "2", "-nps", "2", "--save_probabilities"], check=True)
                import_nnunet(manifest, partition, raw, path, "multiclass", checkpoint=checkpoint, initialization="random")
        if not complete_source(path, manifest, partition, source["sha256"]):
            raise ValueError(f"{name} did not produce complete standardized outputs")
        if read_run(path, manifest, partition)["task"] != "multiclass":
            raise ValueError(f"{name} must provide multiclass probabilities")
        paths[name] = path
    return paths


def plaque_mask(score, coverage, threshold, minimum, affine, lumen=None, distance_mm=None):
    mask = (score >= threshold) & coverage
    if distance_mm is not None:
        if lumen.any():
            distance = ndimage.distance_transform_edt(~lumen, sampling=nib.affines.voxel_sizes(affine))
            mask &= distance <= distance_mm
        else:
            mask[:] = False
    if minimum:
        labels, _ = ndimage.label(mask, C26)
        keep = np.bincount(labels.ravel()) * abs(np.linalg.det(affine[:3, :3])) >= minimum
        keep[0] = False
        mask = keep[labels]
    return mask


def fuse(manifest, partition, roots, archive, record):
    from .evaluation import _csv
    rules = record["rules"]
    destinations = {key: archive / "predictions" / key for key in rules}
    qc = []
    for name, rule in rules.items():
        run_metadata(destinations[name], manifest, partition, "multiclass" if "lumen" in rule else "plaque",
                     model="specialist_pair" if "lumen" in rule else "probability_ensemble",
                     initialization="existing_checkpoints", parameters=rule, candidate_id=record["identities"][name],
                     upstream_checkpoints={m: s["sha256"] for m, s in record["sources"].items()})
    for case, row in select(manifest, partition):
        print(f"Fuse {partition} {case}", flush=True)
        reference = nib.load(row["image"])
        source_data = {key: read_prediction(root, case, reference, "multiclass") for key, root in roots.items()}
        lumen = read_label_prediction(roots["nnunet_resenc"], case, reference, "multiclass") == 1
        for name, rule in rules.items():
            experts = rule["plaque"]
            score = source_data[experts[0]][0][2].copy()
            support = source_data[experts[0]][1].copy()
            for key in experts[1:]:
                score += source_data[key][0][2]
                support &= source_data[key][1]
            if "lumen" in rule:
                support &= source_data[rule["lumen"]][1]
            score /= len(experts)
            score *= support
            plaque = plaque_mask(score, support, rule["threshold"], rule.get("min_mm3", 0),
                                 reference.affine, lumen, rule.get("distance_mm"))
            if "lumen" in rule:
                original = lumen & support
                labels = original.astype(np.uint8)
                labels[plaque] = 2
                remaining = (1 - score) * support
                probs = np.stack([remaining * ~original, remaining * original, score])
                qc.append(dict(case=case, model=name, empty_lumen=not original.any(),
                               lumen_voxels_before=int(original.sum()), lumen_voxels_reassigned=int((original & plaque).sum()),
                               plaque_voxels=int(plaque.sum())))
            else:
                probs = np.stack([(1 - score) * support, score])
                labels = plaque.astype(np.uint8)
            save_prediction(destinations[name], case, reference, probs, support, seg=labels)
    _csv(archive / "specialist_fusion_qc.csv", qc)
    return destinations


def confirmation_tables(reports, out):
    from .evaluation import _csv
    overview, endpoints, aggregates = [], {}, {}
    for name, report in reports.items():
        summary = read_json(report / "summary.json")
        all_rows = read_json(report / "metrics.json")
        all_cases = {kind: {r["case"]: r for r in all_rows if r["branch"] == "ALL"
                            and r["scope"] == "end_to_end" and r["class"] == kind}
                     for kind in ("plaque", "lumen")}
        endpoints[name] = all_cases
        lesions = read_json(report / "lesions.json")
        small = [r for r in lesions if r["volume_mm3"] < 5]
        healthy = summary.get("healthy_branches", {})
        p = list(all_cases["plaque"].values())
        cases = summary["evaluated_ids"]
        branch_rows = [r for r in all_rows if r["branch"] != "ALL" and
                       r["class"] == "plaque" and r["scope"] == "end_to_end" and not r["gt_positive_branch"]]
        aggregates[name] = {
            "lesion_sensitivity": {c: (r["lesion_tp"], r["lesion_tp"] + r["lesion_fn"])
                                   for c, r in all_cases["plaque"].items()},
            "small_lesion_sensitivity": {c: (sum(r["detected"] for r in small if r["case"] == c),
                                             sum(r["case"] == c for r in small)) for c in cases},
            "healthy_branch_fp_rate": {c: (sum(r["pred_positive_branch"] for r in branch_rows if r["case"] == c),
                                           sum(r["case"] == c for r in branch_rows)) for c in cases},
            "lesion_fp_per_patient": {c: (r["lesion_fp"], 1) for c, r in all_cases["plaque"].items()},
        }
        plaque, lumen = (summary.get(key, {}) for key in ("plaque", "lumen"))
        overview.append(dict(candidate=name, cohort_n=summary["cases"],
                             plaque_dice=plaque.get("dice"), plaque_precision=plaque.get("precision"),
                             plaque_recall=plaque.get("recall"), plaque_surface_dice=plaque.get("surface_dice"),
                             lumen_dice=lumen.get("dice"), lumen_surface_dice=lumen.get("surface_dice"),
                             lumen_hd95_mm=lumen.get("hd95_mm"),
                             healthy_branches=healthy.get("n"), healthy_branch_fp_rate=healthy.get("fp_rate"),
                             healthy_branch_fp_components=healthy.get("fp_lesions_per_branch"),
                             lesion_tp=sum(r["lesion_tp"] for r in p), lesion_fp=sum(r["lesion_fp"] for r in p),
                             lesion_fn=sum(r["lesion_fn"] for r in p), small_detected=sum(r["detected"] for r in small),
                             small_total=len(small)))
    paired = []
    for name, values in endpoints.items():
        for kind, comparators in (("plaque", ["nnunet"]), ("lumen", ["nnunet_resenc"])):
            if name == "resenc_lumen_ffr_050_gate3mm":
                comparators = [*comparators, "resenc_lumen_ffr_050"]
            if name == "nnunet_ffr_050_min1" and kind == "plaque":
                comparators = [*comparators, "nnunet_ffr_050"]
            for reference in comparators:
                if reference == name:
                    continue
                first, second = values[kind], endpoints[reference][kind]
                cases = sorted(case for case in set(first) & set(second)
                               if first[case]["dice"] is not None and second[case]["dice"] is not None)
                if not cases:
                    continue
                delta = np.array([first[case]["dice"] - second[case]["dice"] for case in cases])
                draws = np.random.default_rng(SEED).choice(delta, (20000, len(delta)), replace=True).mean(axis=1)
                low, high = np.quantile(draws, [.025, .975])
                paired.append(dict(candidate=name, comparator=reference, endpoint=kind,
                                   paired_n=len(cases), mean_delta=float(delta.mean()),
                                   ci95_low=float(low), ci95_high=float(high), patients_improved=int((delta > 0).sum())))
    _csv(out / "confirmation_metrics.csv", overview)
    _csv(out / "paired_dice_deltas.csv", paired)
    rates = []
    for name, metrics in aggregates.items():
        comparators = ["nnunet"]
        if name == "resenc_lumen_ffr_050_gate3mm":
            comparators.append("resenc_lumen_ffr_050")
        if name == "nnunet_ffr_050_min1":
            comparators.append("nnunet_ffr_050")
        for comparator in comparators:
            if name == comparator:
                continue
            for endpoint, per_case in metrics.items():
                other = aggregates[comparator][endpoint]
                cases = sorted(per_case.keys() & other.keys())
                if not cases:
                    continue
                a, b = (np.array([data[case] for case in cases]) for data in (per_case, other))
                if not a[:, 1].sum() or not b[:, 1].sum():
                    continue
                draw = np.random.default_rng(SEED).integers(len(cases), size=(20000, len(cases)))
                aa, bb = a[draw].sum(axis=1), b[draw].sum(axis=1)
                valid = (aa[:, 1] > 0) & (bb[:, 1] > 0)
                low, high = np.quantile(aa[valid, 0] / aa[valid, 1] - bb[valid, 0] / bb[valid, 1], [.025, .975])
                rates.append(dict(candidate=name, comparator=comparator, endpoint=endpoint, paired_n=len(cases),
                                  delta=float(a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()),
                                  ci95_low=float(low), ci95_high=float(high), bootstrap_draws=int(valid.sum())))
    _csv(out / "paired_detection_deltas.csv", rates)


def confirm(args, manifest):
    from .evaluation import evaluate
    from .protection import guard
    from .reporting import compare
    archive = args.out.resolve() / args.partition
    if not select(manifest, args.partition):
        raise ValueError(f"No {args.partition} patients in the prepared manifest")
    frozen = args.out.resolve() / "frozen_confirmation.json"
    if args.phase == "confirm" and args.partition == "test":
        saved = read_json(frozen)
        if any(getattr(args, m) for m in MODELS) or args.nnunet_results:
            raise ValueError("Confirmation uses the checkpoints recorded by the freeze command")
        choices = {key: Path(row["checkpoint"]) for key, row in saved["sources"].items()}
    else:
        saved = None
        choices = checkpoint_choices(args.bench, {m: getattr(args, m) for m in MODELS}, args.nnunet_results)
    sources = sources_for(manifest, args.partition, archive, choices, args.bench)
    record = confirmation_record(args.dataset, manifest, sources, saved["decision"] if saved else args.decision)
    if args.phase == "freeze":
        if args.partition != "test":
            raise ValueError("Freeze requires --partition test")
        write_json(frozen, record, lock=True)
        print("Frozen confirmation:", frozen)
        return
    if args.partition == "test":
        if record != saved:
            raise ValueError("Dataset, checkpoint, plans, rules, or benchmark source changed after freeze")
        for candidate_id in record["candidate_ids"]:
            guard(manifest, args.partition, frozen, candidate_id)
    write_json(archive / "protocol.json", record, lock=True)
    roots = obtain_sources(manifest, args.partition, archive, sources, args.device, args.nnunet_python)
    roots.update(fuse(manifest, args.partition, roots, archive, record))
    reports = {}
    for name, path in roots.items():
        report = archive / "reports" / name
        evaluate(manifest, args.partition, path, report, **PROTOCOL)
        reports[name] = report
    compare(list(reports.values()), archive / "comparison")
    confirmation_tables(reports, archive / "comparison")
    print("Confirmation reports:", archive / "comparison")

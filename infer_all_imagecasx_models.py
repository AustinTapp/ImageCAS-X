"""Batch inference over a folder of CCTA volumes using ImageCAS-X models.

Designed for:
    repo:        D:\\Code\\ImageCAS-X
    checkpoints: D:\\Code\\ImageCAS-X\\model_ckpts
    input:       D:\\Code\\ImageCAS-X\\image_input
    output:      D:\\Code\\ImageCAS-X\\image_results

Default behavior runs every FINAL ImageCAS-X method that can be inferred directly
from a CCTA volume and whose checkpoint(s) can be identified:
    - CAS-Net
    - FFR-UNet
    - Swin UNETR
    - ImageCAS baseline (requires all five stage checkpoint fields)

ADE-HTL is intentionally not run from image-only input because its final inference
pipeline requires TotalSegmentator heartchambers_highres masks / ADE auxiliary data.
The repository's external nnU-Net, nnU-Net+clDice, and TotalSegmentator methods are
also not loaded by ImageCAS-X's internal model registry.

Use --include-stages if you also want raw-CT-compatible intermediate coarse models
(ImageCAS stage 1, ImageCAS stage 2, ADE-HTL stage 1) written as separate outputs.
These are intermediate predictions, not final benchmark segmentations.

Each model writes one NIfTI mask per input scan to:
    <output_dir>/<model_name>/<case_id>.nii.gz

A run_summary.json records checkpoint discovery, successes, skips, and failures.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


DEFAULT_REPO_ROOT = Path(r"D:\Code\ImageCAS-X")
DEFAULT_CHECKPOINT_DIR = DEFAULT_REPO_ROOT / "model_ckpts"
DEFAULT_INPUT_DIR = DEFAULT_REPO_ROOT / "image_input"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "image_results"

MEDICAL_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")
CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".ckpt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all available ImageCAS-X CCTA segmentation models over a folder."
    )
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--expected-cases",
        type=int,
        default=9,
        help="Require this many input volumes. Set 0 to disable the count check (default: 9).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc. (default: auto)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute predictions that already exist.",
    )
    p.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable the repository's default mirror test-time augmentation.",
    )
    p.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Override patch inference batch size for patch-trained models.",
    )
    p.add_argument(
        "--include-stages",
        action="store_true",
        help="Also run raw-CT-compatible intermediate ImageCAS/ADE coarse stages.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover input scans/checkpoints and report what would run.",
    )
    return p.parse_args()


def import_runtime(repo_root: Path) -> SimpleNamespace:
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")
    if not (repo_root / "inference.py").exists():
        raise FileNotFoundError(
            f"Could not find ImageCAS-X inference.py under repository root: {repo_root}"
        )

    sys.path.insert(0, str(repo_root))

    try:
        import numpy as np
        import SimpleITK as sitk
        import torch
        from utils.config import BenchmarkConfig
        from utils import io as bio
        from models.registry import build_model
        from preprocessing.pipeline import build_preprocessing
        from postprocessing.pipeline import build_postprocessing
        from inference import run_inference, _resample_probs_to_original_space
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the ImageCAS-X runtime. Run this from the Python environment "
            "where the cloned repository and its dependencies are installed (pip install -e .)."
        ) from exc

    return SimpleNamespace(
        np=np,
        sitk=sitk,
        torch=torch,
        BenchmarkConfig=BenchmarkConfig,
        bio=bio,
        build_model=build_model,
        build_preprocessing=build_preprocessing,
        build_postprocessing=build_postprocessing,
        run_inference=run_inference,
        resample_probs=_resample_probs_to_original_space,
    )


def normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def strip_medical_extension(name: str) -> str:
    low = name.lower()
    for ext in sorted(MEDICAL_EXTENSIONS, key=len, reverse=True):
        if low.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(x) if x.isdigit() else x for x in parts]


def discover_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and any(p.name.lower().endswith(ext) for ext in MEDICAL_EXTENSIONS)
    ]
    images.sort(key=natural_key)

    case_ids = [strip_medical_extension(p.name) for p in images]
    duplicates = sorted({x for x in case_ids if case_ids.count(x) > 1})
    if duplicates:
        raise RuntimeError(
            "Duplicate case IDs after stripping medical-image extensions: " + ", ".join(duplicates)
        )
    return images


def discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    files = [
        p
        for p in checkpoint_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p).lower())


def _checkpoint_score(path: Path, checkpoint_root: Path, aliases: list[str]) -> int:
    try:
        rel = path.relative_to(checkpoint_root)
    except ValueError:
        rel = path

    rel_norm = normalise_name(str(rel))
    stem_norm = normalise_name(path.stem)
    best = 0

    for alias in aliases:
        a = normalise_name(alias)
        if not a:
            continue
        if stem_norm == a:
            best = max(best, 120)
        if stem_norm in (f"{a}_best", f"best_{a}"):
            best = max(best, 118)
        if stem_norm.startswith(f"{a}_best"):
            best = max(best, 116)
        if a in stem_norm:
            best = max(best, 105)
        if a in rel_norm:
            best = max(best, 95)

        tokens = [t for t in a.split("_") if len(t) > 1]
        if tokens and all(t in rel_norm.split("_") for t in tokens):
            best = max(best, 70 + min(len(tokens), 20))

    return best


def resolve_checkpoint(
    checkpoints: list[Path], checkpoint_root: Path, aliases: list[str]
) -> tuple[Path | None, list[tuple[int, Path]]]:
    ranked = []
    for p in checkpoints:
        score = _checkpoint_score(p, checkpoint_root, aliases)
        if score > 0:
            ranked.append((score, p))
    ranked.sort(key=lambda x: (-x[0], len(str(x[1])), str(x[1]).lower()))
    return (ranked[0][1] if ranked else None), ranked[:5]


def pretty_candidates(ranked: list[tuple[int, Path]], checkpoint_root: Path) -> list[str]:
    out = []
    for score, path in ranked:
        try:
            rel = path.relative_to(checkpoint_root)
        except ValueError:
            rel = path
        out.append(f"{rel} [score={score}]")
    return out


def checkpoint_aliases() -> dict[str, list[str]]:
    # Aliases are matched against both the filename and its relative folder path.
    return {
        "cas_net": ["cas_net", "casnet", "cas_net_best"],
        "ffr_unet": ["ffr_unet", "ffrunet", "ffr_unet_best"],
        "swin_unetr": ["swin_unetr", "swinunetr", "swin_unetr_best"],
        "imagecas_stage1_coarse": [
            "imagecas_stage1_coarse",
            "imagecas_stage_1_coarse",
            "imagecas_coarse_stage1",
        ],
        "imagecas_stage2_coarse_dilated": [
            "imagecas_stage2_coarse_dilated",
            "imagecas_stage_2_coarse_dilated",
            "imagecas_stage2_dilated",
        ],
        "imagecas_stage3_patch_16": [
            "imagecas_stage3_patch_16",
            "imagecas_stage3_patch16",
            "imagecas_patch_16",
        ],
        "imagecas_stage3_patch_32": [
            "imagecas_stage3_patch_32",
            "imagecas_stage3_patch32",
            "imagecas_patch_32",
        ],
        "imagecas_stage3_patch_64": [
            "imagecas_stage3_patch_64",
            "imagecas_stage3_patch64",
            "imagecas_patch_64",
        ],
        "ade_htl_stage1_coarse": [
            "ade_htl_stage1_coarse",
            "ade_htl_stage_1_coarse",
            "ade_stage1_coarse",
        ],
        "ade_htl": ["ade_htl_best", "ade_htl_final", "ade_htl"],
    }


def discover_model_plan(
    repo_root: Path,
    checkpoint_root: Path,
    checkpoints: list[Path],
    include_stages: bool,
) -> tuple[list[dict], dict, dict]:
    aliases = checkpoint_aliases()
    resolved: dict[str, Path | None] = {}
    candidates: dict[str, list[str]] = {}

    for key, alias_list in aliases.items():
        ckpt, ranked = resolve_checkpoint(checkpoints, checkpoint_root, alias_list)
        resolved[key] = ckpt
        candidates[key] = pretty_candidates(ranked, checkpoint_root)

    plan: list[dict] = []
    skipped: dict[str, str] = {}

    simple_final = [
        ("cas_net", "configs/cas_net.json", "cas_net"),
        ("ffr_unet", "configs/ffr_unet.json", "ffr_unet"),
        ("swin_unetr", "configs/swin_unetr.json", "swin_unetr"),
    ]
    for output_name, config_rel, ckpt_key in simple_final:
        cfg_path = repo_root / config_rel
        ckpt = resolved.get(ckpt_key)
        if not cfg_path.exists():
            skipped[output_name] = f"config not found: {cfg_path}"
        elif ckpt is None:
            skipped[output_name] = "no matching checkpoint found"
        else:
            plan.append(
                {
                    "name": output_name,
                    "config": cfg_path,
                    "checkpoint": ckpt,
                    "staged_checkpoints": None,
                    "kind": "final",
                }
            )

    imagecas_keys = [
        "imagecas_stage1_coarse",
        "imagecas_stage2_coarse_dilated",
        "imagecas_stage3_patch_16",
        "imagecas_stage3_patch_32",
        "imagecas_stage3_patch_64",
    ]
    imagecas_missing = [k for k in imagecas_keys if resolved.get(k) is None]
    imagecas_cfg = repo_root / "configs/imagecas_inference.json"
    if not imagecas_cfg.exists():
        skipped["imagecas_baseline"] = f"config not found: {imagecas_cfg}"
    elif imagecas_missing:
        skipped["imagecas_baseline"] = (
            "missing staged checkpoint(s): " + ", ".join(imagecas_missing)
        )
    else:
        plan.append(
            {
                "name": "imagecas_baseline",
                "config": imagecas_cfg,
                "checkpoint": None,
                "staged_checkpoints": {
                    "coarse_checkpoint": resolved["imagecas_stage1_coarse"],
                    "dilated_checkpoint": resolved["imagecas_stage2_coarse_dilated"],
                    "patch_checkpoint_16": resolved["imagecas_stage3_patch_16"],
                    "patch_checkpoint_32": resolved["imagecas_stage3_patch_32"],
                    "patch_checkpoint_64": resolved["imagecas_stage3_patch_64"],
                },
                "kind": "final",
            }
        )

    # Final ADE-HTL is not image-only. Record this explicitly instead of running an
    # anatomically incomplete approximation.
    if resolved.get("ade_htl") is not None:
        skipped["ade_htl"] = (
            "final ADE-HTL checkpoint detected, but final inference requires "
            "TotalSegmentator heartchambers_highres / ADE auxiliary inputs; image_input alone is insufficient"
        )
    else:
        skipped["ade_htl"] = (
            "not scheduled: final ADE-HTL requires TotalSegmentator heartchambers_highres / ADE auxiliary inputs"
        )

    if include_stages:
        stage_specs = [
            (
                "imagecas_stage1_coarse",
                "configs/imagecas_stage1_coarse.json",
                "imagecas_stage1_coarse",
            ),
            (
                "imagecas_stage2_coarse_dilated",
                "configs/imagecas_stage2_coarse_dilated.json",
                "imagecas_stage2_coarse_dilated",
            ),
            (
                "ade_htl_stage1_coarse",
                "configs/ade_htl_stage1_coarse.json",
                "ade_htl_stage1_coarse",
            ),
        ]
        for output_name, config_rel, ckpt_key in stage_specs:
            cfg_path = repo_root / config_rel
            ckpt = resolved.get(ckpt_key)
            if not cfg_path.exists():
                skipped[output_name] = f"config not found: {cfg_path}"
            elif ckpt is None:
                skipped[output_name] = "no matching checkpoint found"
            else:
                plan.append(
                    {
                        "name": output_name,
                        "config": cfg_path,
                        "checkpoint": ckpt,
                        "staged_checkpoints": None,
                        "kind": "intermediate_stage",
                    }
                )

    return plan, skipped, candidates


def configure_model(rt: SimpleNamespace, spec: dict, args: argparse.Namespace):
    cfg = rt.BenchmarkConfig.from_json(str(spec["config"]))

    # We deliberately do not call cfg.validate(), because this runner does not use
    # the repository Dataset/FileList machinery. It reads the input CCTAs directly.
    if spec["staged_checkpoints"]:
        for field, path in spec["staged_checkpoints"].items():
            setattr(cfg.model, field, str(path))
    else:
        cfg.model.checkpoint = str(spec["checkpoint"])

    if args.no_tta:
        cfg.data.params["inference_mirror_tta"] = False
    if args.inference_batch_size is not None:
        if args.inference_batch_size < 1:
            raise ValueError("--inference-batch-size must be >= 1")
        cfg.data.params["inference_batch_size"] = int(args.inference_batch_size)

    return cfg


def choose_device(rt: SimpleNamespace, requested: str):
    requested = requested.strip().lower()
    if requested == "auto":
        requested = "cuda:0" if rt.torch.cuda.is_available() else "cpu"
    device = rt.torch.device(requested)
    if device.type == "cuda" and not rt.torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}) but torch.cuda.is_available() is False")
    return device


def prediction_path(model_out_dir: Path, input_path: Path) -> Path:
    case_id = strip_medical_extension(input_path.name)
    return model_out_dir / f"{case_id}.nii.gz"


def infer_case(
    rt: SimpleNamespace,
    model,
    cfg,
    preprocessing,
    postprocessing,
    device,
    image_path: Path,
    out_path: Path,
):
    # Keep the original geometry for the final output. The preprocessing pipeline may
    # replace sample['sitk_img'] with a resampled image.
    volume, reference_img = rt.bio.load_volume(str(image_path))
    case_id = strip_medical_extension(image_path.name)

    sample = {
        "volume": volume.astype(rt.np.float32, copy=False),
        "scan_id": case_id,
        "spacing": tuple(float(v) for v in reference_img.GetSpacing()),
        "sitk_img": reference_img,
    }
    sample = preprocessing(sample)

    processed = rt.np.ascontiguousarray(sample["volume"], dtype=rt.np.float32)
    volume_tensor = rt.torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
    processed_spacing = tuple(float(v) for v in sample["spacing"])

    # Reuse ImageCAS-X's own full-volume strategy: direct forward for volume models,
    # or sliding-window stitching for patch/random-crop models, including mirror TTA.
    output = rt.run_inference(model, volume_tensor, cfg, device, scan_id=case_id)
    logits = output["logits"][0]
    prob = rt.torch.sigmoid(logits).cpu().numpy()

    # All models scheduled by this image-only runner produce mask/mask-sigmoid outputs.
    # ADE-HTL's 27-channel connectivity output is intentionally excluded above because
    # its required auxiliary inputs are not available from image_input alone.
    prob_orig = rt.resample_probs(prob, processed_spacing, reference_img)
    pred_sample = postprocessing({"pred": prob_orig, "scan_id": case_id})
    pred_orig = pred_sample["mask"]

    rt.bio.save_mask(pred_orig, reference_img, str(out_path))

    del volume_tensor, logits, prob, prob_orig, pred_orig, sample


def run_model(
    rt: SimpleNamespace,
    spec: dict,
    images: list[Path],
    output_root: Path,
    device,
    args: argparse.Namespace,
) -> dict:
    model_name = spec["name"]
    print("\n" + "=" * 88)
    print(f"MODEL: {model_name}")
    print(f"CONFIG: {spec['config']}")
    if spec["staged_checkpoints"]:
        for key, value in spec["staged_checkpoints"].items():
            print(f"  {key}: {value}")
    else:
        print(f"CHECKPOINT: {spec['checkpoint']}")
    print("=" * 88)

    cfg = configure_model(rt, spec, args)
    preprocessing = rt.build_preprocessing(cfg)
    postprocessing = rt.build_postprocessing(cfg)

    model = rt.build_model(cfg).to(device)
    model.eval()

    model_out_dir = output_root / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    errors: dict[str, str] = {}
    n_forwarded = 0

    for index, image_path in enumerate(images, start=1):
        out_path = prediction_path(model_out_dir, image_path)
        case_id = strip_medical_extension(image_path.name)

        if out_path.exists() and not args.overwrite:
            print(f"[{index:02d}/{len(images):02d}] {case_id}: exists -> skip")
            continue

        print(f"[{index:02d}/{len(images):02d}] {case_id}: infer ...", flush=True)
        case_start = time.perf_counter()
        try:
            infer_case(
                rt,
                model,
                cfg,
                preprocessing,
                postprocessing,
                device,
                image_path,
                out_path,
            )
            n_forwarded += 1
            elapsed = time.perf_counter() - case_start
            print(f"             saved -> {out_path}  ({elapsed:.2f} s)")
        except Exception as exc:
            errors[case_id] = f"{type(exc).__name__}: {exc}"
            print(f"             FAILED: {errors[case_id]}")
            traceback.print_exc()

        if device.type == "cuda":
            rt.torch.cuda.empty_cache()
        gc.collect()

    expected_paths = [prediction_path(model_out_dir, p) for p in images]
    missing = [str(p) for p in expected_paths if not p.exists()]
    produced = len(expected_paths) - len(missing)
    elapsed_total = time.perf_counter() - started

    result = {
        "status": "ok" if not missing and not errors else "incomplete",
        "kind": spec["kind"],
        "config": str(spec["config"]),
        "checkpoint": str(spec["checkpoint"]) if spec["checkpoint"] else None,
        "staged_checkpoints": (
            {k: str(v) for k, v in spec["staged_checkpoints"].items()}
            if spec["staged_checkpoints"]
            else None
        ),
        "output_dir": str(model_out_dir),
        "expected_predictions": len(images),
        "predictions_present": produced,
        "newly_forwarded": n_forwarded,
        "missing_predictions": missing,
        "errors": errors,
        "elapsed_seconds": elapsed_total,
    }

    print(
        f"[{model_name}] predictions present: {produced}/{len(images)} "
        f"| newly forwarded: {n_forwarded} | elapsed: {elapsed_total:.1f} s"
    )

    del model
    if device.type == "cuda":
        rt.torch.cuda.empty_cache()
    gc.collect()
    return result


def serialise_plan(plan: list[dict]) -> list[dict]:
    out = []
    for spec in plan:
        out.append(
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "config": str(spec["config"]),
                "checkpoint": str(spec["checkpoint"]) if spec["checkpoint"] else None,
                "staged_checkpoints": (
                    {k: str(v) for k, v in spec["staged_checkpoints"].items()}
                    if spec["staged_checkpoints"]
                    else None
                ),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    images = discover_inputs(input_dir)
    if args.expected_cases > 0 and len(images) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} input volume(s), but found {len(images)} in {input_dir}.\n"
            + "Found: "
            + ", ".join(p.name for p in images)
        )
    if not images:
        raise RuntimeError(f"No supported medical-image files found in {input_dir}")

    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise RuntimeError(f"No .pt/.pth/.ckpt files found under {checkpoint_dir}")

    plan, skipped, candidates = discover_model_plan(
        repo_root, checkpoint_dir, checkpoints, args.include_stages
    )

    print(f"Repository:  {repo_root}")
    print(f"Checkpoints: {checkpoint_dir} ({len(checkpoints)} files discovered)")
    print(f"Input:       {input_dir} ({len(images)} cases)")
    print(f"Output:      {output_dir}")
    print("\nInput cases:")
    for p in images:
        print(f"  - {p.name}")

    print("\nResolved model plan:")
    if plan:
        for spec in plan:
            print(f"  RUN  {spec['name']} ({spec['kind']})")
            if spec["checkpoint"]:
                print(f"       checkpoint: {spec['checkpoint']}")
            else:
                for k, v in spec["staged_checkpoints"].items():
                    print(f"       {k}: {v}")
    else:
        print("  (none)")

    print("\nNot scheduled / unavailable:")
    for name, reason in sorted(skipped.items()):
        print(f"  SKIP {name}: {reason}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    summary = {
        "repo_root": str(repo_root),
        "checkpoint_dir": str(checkpoint_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "expected_cases": args.expected_cases,
        "input_cases": [p.name for p in images],
        "checkpoint_files": [str(p) for p in checkpoints],
        "checkpoint_candidates": candidates,
        "plan": serialise_plan(plan),
        "skipped": skipped,
        "results": {},
        "notes": [
            "Final ADE-HTL is excluded from image-only inference because the repository requires TotalSegmentator heartchambers_highres / ADE auxiliary inputs.",
            "nnU-Net, nnU-Net+clDice, and TotalSegmentator are external methods in ImageCAS-X and are not loaded by the repository's internal model registry.",
        ],
    }

    if args.dry_run:
        summary["dry_run"] = True
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nDry run complete. Summary -> {summary_path}")
        return 0

    if not plan:
        summary["fatal_error"] = "No runnable model checkpoints could be resolved."
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nNo runnable models resolved. Inspect checkpoint_candidates in {summary_path}")
        return 2

    rt = import_runtime(repo_root)
    device = choose_device(rt, args.device)
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU:    {rt.torch.cuda.get_device_name(device)}")

    overall_start = time.perf_counter()
    for spec in plan:
        try:
            summary["results"][spec["name"]] = run_model(
                rt, spec, images, output_dir, device, args
            )
        except Exception as exc:
            print(f"\n[FATAL FOR MODEL {spec['name']}] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary["results"][spec["name"]] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["total_elapsed_seconds"] = time.perf_counter() - overall_start
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("FINAL COUNTS")
    print("=" * 88)
    all_complete = True
    for spec in plan:
        result = summary["results"].get(spec["name"], {})
        present = result.get("predictions_present", 0)
        expected = result.get("expected_predictions", len(images))
        status = result.get("status", "failed")
        print(f"{spec['name']:<36} {present:>2}/{expected:<2}  {status}")
        if status != "ok" or present != expected:
            all_complete = False

    print(f"\nSummary written to: {summary_path}")
    if all_complete:
        print(
            f"SUCCESS: every scheduled model has exactly {len(images)} prediction(s) present."
        )
        return 0

    print("WARNING: one or more scheduled models did not produce every expected prediction.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

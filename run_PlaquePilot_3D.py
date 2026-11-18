"""Matched 3D plaque experiments on the approved, historical PlaquePilot split."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.metadata
import math
from pathlib import Path
import shutil
import sys

from plaque_benchmark.data import digest, file_sha256, read_json, select, write_json

ROOT = Path(__file__).resolve().parent
MODELS = ("ffr_unet", "swin_unetr_v2", "mednext")
PARAMS = {"ffr_unet": {}, "swin_unetr_v2": {"feature_size": 24, "use_checkpoint": True},
          "mednext": {"size": "S", "kernel_size": 3}}
MANIFEST_SHA = "7499e140a35ca8e6d2a526cbfcde13ee90987a9cf296d3677defe4c172f8ea27"
SPLIT_SHA = "3a6bd0785d0d4c463d00d8f72165e8f846cc0d1d6c35209c11905104a1e880bb"
PHASES = ("check", "init", "tune", "train", "score", "extend", "run", "all", "freeze", "test", "legacy-val")


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, data):
        self.stream.write(data)
        if not self.log.closed:
            self.log.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        if not self.log.closed:
            self.log.flush()

    def isatty(self):
        return self.stream.isatty()


@contextmanager
def logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            try:
                yield
            except Exception:
                import traceback
                traceback.print_exc(file=stream)
                raise


def case_hashes(cases):
    def case_hash(item):
        case, row = item
        ct = file_sha256(row["image"])
        if ct != row["image_sha256"]:
            raise ValueError(f"Archived CT differs for {case}")
        paths = dict(label=row["label"], branch_lumen=row["branch_lumen"], **row["branches"])
        return case, dict(ct=ct, **{k: file_sha256(v) for k, v in paths.items()})

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(case_hash, cases))


def audit(bench, out, manifest, protocol, device):
    from ctaug import StepTransform
    import ctaug
    import optuna
    import torch
    from monai.networks.nets import SwinUNETR
    from inspect import signature

    if "use_v2" not in signature(SwinUNETR).parameters:
        raise ValueError("The installed MONAI SwinUNETR constructor lacks use_v2.")
    import einops

    dataset = bench / "prepared" / "dataset.json"
    if file_sha256(dataset) != MANIFEST_SHA or manifest["split_sha256"] != SPLIT_SHA:
        raise ValueError("Use the original PlaquePilot_confirmed prepared/dataset.json. Its archived identity differs.")
    partitions = {p: [c for c, _ in select(manifest, p)] for p in ("train", "val", "test")}
    if [len(v) for v in partitions.values()] != [75, 16, 16]:
        raise ValueError("Expected the approved 75/16/16 patient split.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the active environment.")
    # Validate the original derived assets against the existing approval record.
    receipt = read_json(bench / "approval_receipt.json")
    if receipt["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("Approval receipt and archived manifest differ.")
    for asset in receipt["derived_assets"]:
        st = Path(asset["path"]).stat()
        if (st.st_size, st.st_mtime_ns) != (asset["size"], asset["mtime_ns"]):
            raise ValueError(f"Prepared asset differs from the original approval receipt: {asset['path']}")
    print("Checking CT and segmentation content hashes for the training and validation patients.", flush=True)

    assets = case_hashes(select(manifest, "train") + select(manifest, "val"))
    source = [Path(__file__).resolve(), ROOT / "inference.py"]
    for folder in ("plaque_benchmark", "models", "utils"):
        source.extend(sorted((ROOT / folder).rglob("*.py")))
    versions = {k: importlib.metadata.version(k) for k in ("torch","monai","optuna","numpy","nibabel","scipy","einops")}
    ctaug_root = Path(ctaug.__file__).parent
    context = dict(manifest_sha256=MANIFEST_SHA, split_sha256=SPLIT_SHA, partitions=partitions,
                   files=assets, protocol=protocol, model_params=PARAMS, device=device,
                   gpu=torch.cuda.get_device_name() if device=="cuda" else None,
                   versions=versions, python=sys.version, cuda=torch.version.cuda,
                   ctaug_source={p.relative_to(ctaug_root).as_posix():file_sha256(p) for p in sorted(ctaug_root.rglob("*.py"))},
                   source={p.relative_to(ROOT).as_posix():file_sha256(p) for p in source})
    check_comparability(protocol)
    write_json(out / "protocol_lock.json", context, lock=True)
    print(f"Verified 75 train / 16 validation / 16 held out. Protocol: {out / 'protocol_lock.json'}", flush=True)


def configuration(protocol, model, stage, optimizer="adamw", lr=None):
    cfg = dict(protocol["shared"], model=model, model_params=PARAMS[model], optimizer=optimizer)
    if lr is not None:
        cfg["lr"] = lr
    if stage in ("init", "tune"):
        cfg.update(epochs=1 if stage=="init" else protocol["search_epochs"], patience=0, early_stopping_start_epoch=0)
    return cfg


def check_comparability(protocol):
    shared = protocol["shared"]
    if not (0 < protocol["comparison_epochs"] == shared["early_stopping_start_epoch"] < shared["epochs"]):
        raise ValueError("The fixed comparison must precede the early-stopping window.")
    if min(shared["patience"], protocol["search_epochs"]) < 1:
        raise ValueError("Invalid search budget or extension patience.")
    grid = protocol["optimizer_grid"]
    if set(grid) != {"adamw","sgd_nesterov"} or len({len(v) for v in grid.values()}) != 1:
        raise ValueError("Both optimizer families require the same number of LR options.")
    for stage in ("init","tune","train"):
        reference = None
        for model in MODELS:
            for opt, rates in grid.items():
                for lr in rates:
                    cfg = configuration(protocol,model,stage,opt,lr)
                    common = {k:v for k,v in cfg.items() if k not in {"model","model_params","optimizer","lr"}}
                    if reference is not None and common != reference:
                        raise ValueError(f"Network-dependent change to shared settings in {stage}.")
                    reference = common


def fit(manifest, cfg, folder, device, target_epoch, *, allow_stop=False):
    import torch
    from plaque_benchmark.native import train

    write_json(folder / "override.json", cfg, lock=True)
    model_dir = folder / "model"
    last = model_dir / "last.pt"
    with logging(folder / "training.log"):
        train(manifest, "multiclass", cfg, model_dir, device=device,
              resume=last.is_file(), stop_after_epoch=target_epoch)
    state = torch.load(last, map_location="cpu", weights_only=True)
    history = state["history"]
    epoch = state["epoch"]
    if epoch != target_epoch and not (allow_stop and epoch >= cfg["early_stopping_start_epoch"] and
                                       state["stalled"] >= cfg["patience"]):
        raise RuntimeError(f"Expected {target_epoch} epochs. Found {epoch} in {folder}")
    draws = cfg["steps"] * cfg["batch_size"] * cfg["accumulate"]
    if len(history) != epoch:
        raise RuntimeError(f"Missing epoch history in {folder}")
    for i, row in enumerate(history, 1):
        if (row["epoch"] != i or row["patches"] != i * draws or row["optimizer_steps"] != i * cfg["steps"]
            or not all(math.isfinite(row[key]) for key in ("train_loss","val_patch_loss","val_patch_dice_plaque"))):
            raise RuntimeError(f"Invalid metrics or exposure at {folder}, epoch {i}")
    return history


def init_all(manifest, protocol, out, device):
    import torch
    for model in MODELS:
        folder = out / "init" / model
        if not (folder / "PASS.json").is_file():
            cfg = configuration(protocol, model, "init")
            fit(manifest, cfg, folder, device, 1)
            best = torch.load(folder / "model/best.pt", map_location="cpu", weights_only=True)
            if best["epoch"] != 1 or not best["model"]:
                raise RuntimeError(f"Unreadable model weights for {model}")
            write_json(folder / "PASS.json", {"last":file_sha256(folder/"model/last.pt"),
                       "best":file_sha256(folder/"model/best.pt")}, lock=True)
        check_init(out, model)
        print(f"Initialization PASS: {model}", flush=True)


def check_init(out, model):
    folder = out / "init" / model
    if not (folder / "PASS.json").is_file():
        raise RuntimeError(f"Run init first. Missing successful initialization for {model}.")
    receipt = read_json(folder / "PASS.json")
    for name in ("last","best"):
        if file_sha256(folder / "model" / (name+".pt")) != receipt[name]:
            raise ValueError(f"Initialization checkpoint changed for {model}")


def tune(manifest, protocol, out, device):
    import optuna
    for model in MODELS:
        check_init(out, model)
    for model in MODELS:
        folder = out / "search" / model
        grid = protocol["optimizer_grid"]
        space = {"optimizer":list(grid), "lr_index":list(range(len(next(iter(grid.values())))))}
        sampler = optuna.samplers.GridSampler(space, seed=protocol["shared"]["seed"])
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            optimizer = trial.suggest_categorical("optimizer", space["optimizer"])
            index = trial.suggest_categorical("lr_index", space["lr_index"])
            lr = grid[optimizer][index]
            trial.set_user_attr("lr", lr)
            cfg = configuration(protocol, model, "tune", optimizer, lr)
            path = folder / f"{optimizer}_{index}"
            result = path / "trial_result.json"
            if result.exists():
                saved = read_json(result)
                if saved["config"] != cfg or saved["completed_epochs"] != cfg["epochs"]:
                    raise ValueError(f"Search receipt differs in {path}")
                return saved["objective"]
            history = fit(manifest, cfg, path, device, cfg["epochs"])
            score = max(row["val_patch_dice_plaque"] for row in history)
            write_json(result, dict(config=cfg, objective=score, completed_epochs=len(history)), lock=True)
            return score

        study.optimize(objective, n_trials=sum(map(len,grid.values())), n_jobs=1)
        rows = [dict(trial=t.number, optimizer=t.params["optimizer"], lr=t.user_attrs["lr"],
                     lr_index=t.params["lr_index"], score=t.value) for t in study.trials]
        if len({(r["optimizer"],r["lr"]) for r in rows}) != sum(map(len,grid.values())):
            raise RuntimeError("Incomplete Optuna grid.")
        from plaque_benchmark.evaluation import _csv
        _csv(folder/"trials.csv", rows)
        # Stable tie-break across Optuna versions.
        winner = min(rows, key=lambda r: (-r["score"], r["optimizer"], r["lr"]))
        write_json(folder/"winner.json", winner, lock=True)
        write_json(folder/"best_config.json", configuration(protocol, model, "train",
                   winner["optimizer"], winner["lr"]), lock=True)
        print(f"Selected {model}: {winner}", flush=True)


def selected_config(out, protocol, model):
    winner = read_json(out / "search" / model / "winner.json")
    return configuration(protocol, model, "train", winner["optimizer"], winner["lr"])


def seal_checkpoint(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(".tmp")
        shutil.copyfile(source, temp)
        temp.replace(target)
    if file_sha256(target) != file_sha256(source):
        raise ValueError(f"Sealed checkpoint mismatch: {target}")


def verify_receipt(folder):
    receipt = read_json(folder / "stage.json")
    checkpoint = folder / "best.pt"
    if file_sha256(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError(f"Archived weights changed at {folder}")
    return checkpoint


def train_stage(manifest, protocol, out, device, models, stage):
    for model in MODELS:
        check_init(out,model)
        selected_config(out,protocol,model)
    for model in models:
        archive = out / stage / model
        if (archive / "stage.json").exists():
            verify_receipt(archive)
            continue
        cfg = selected_config(out, protocol, model)
        training = out / "training" / model
        write_json(training/"specification.json", {"context":{"configuration":cfg,
                   "dataset_sha256":MANIFEST_SHA,
                   "environment":{"device":device}}}, lock=True)
        target = protocol["comparison_epochs"] if stage=="matched" else cfg["epochs"]
        if stage=="long":
            verify_receipt(out / "matched" / model)
            last = training/"model/last.pt"
            if not last.exists():
                raise FileNotFoundError("Extension needs the live last.pt at the matched boundary.")
            start = archive/"continuation.json"
            original = read_json(out/"matched"/model/"stage.json")["last_sha256"]
            if not start.exists() and file_sha256(last) != original:
                raise ValueError(f"Live checkpoint differs from the matched endpoint for {model}")
            write_json(start, {"initial_last_sha256":original}, lock=True)
        history = fit(manifest, cfg, training, device, target, allow_stop=stage=="long")
        seal_checkpoint(training/"model/best.pt", archive/"best.pt")
        best = max(history, key=lambda row:row["val_patch_dice_plaque"])
        shutil.copyfile(training/"model/history.csv", archive/"history.csv")
        write_json(archive/"stage.json", dict(checkpoint_sha256=file_sha256(archive/"best.pt"),
                   last_sha256=file_sha256(training/"model/last.pt"),
                   best_epoch=best["epoch"], epochs=len(history), patches=history[-1]["patches"],
                   updates=history[-1]["optimizer_steps"], epoch_hours=sum(r["seconds"] for r in history)/3600),lock=True)


def score(manifest, out, device, partition, stage, models, frozen=None):
    from plaque_benchmark.native import infer
    from plaque_benchmark.evaluation import evaluate, _csv
    from plaque_benchmark.protection import guard
    from plaque_benchmark.reporting import compare
    from plaque_benchmark.round2_confirm import complete_source

    root = out / "scoring" / partition / stage
    rows, reports = [], []
    for model in models:
        folder = out / stage / model
        checkpoint = verify_receipt(folder)
        receipt = read_json(folder / "stage.json")
        sha = receipt["checkpoint_sha256"]
        guard(manifest,partition,frozen,sha)
        predictions = root/"predictions"/model
        report = root/"reports"/model
        # The summary is written after per-case and lesion metrics.
        if not (report/"summary.json").is_file():
            finished = predictions / "inference_complete.json"
            if not finished.is_file() or not complete_source(predictions,manifest,partition,sha):
                with logging(root/"logs"/f"{model}.log"):
                    infer(manifest,partition,checkpoint,predictions,device=device)
            if not complete_source(predictions,manifest,partition,sha):
                raise RuntimeError(f"Incomplete predictions for {model}")
            write_json(finished, {"checkpoint_sha256":sha,"cases":[c for c,_ in select(manifest,partition)]}, lock=True)
            evaluate(manifest,partition,predictions,report,surface_tolerance=1.,lesion_iou=0.)
        summary=read_json(report/"summary.json")
        if (summary["run"]["checkpoint_sha256"] != sha or summary["run"]["partition"] != partition
            or summary["run"]["split_sha256"] != manifest["split_sha256"]
            or summary["evaluated_ids"] != [c for c,_ in select(manifest,partition)]):
            raise ValueError(f"Incompatible report at {report}")
        plaque, lumen, healthy = (summary.get(k,{}) for k in ("plaque","lumen","healthy_branches"))
        rows.append(dict(model=model, plaque_dice=plaque["dice"], lumen_dice=lumen["dice"],
                         plaque_precision=plaque["precision"], plaque_recall=plaque["recall"],
                         healthy_branch_fp_rate=healthy.get("fp_rate"), cases=summary["cases"],
                         plaque_dice_n=plaque["dice_defined_n"], **receipt))
        reports.append(report)
    compare(reports,root/"comparison")
    if stage == "matched":
        paired_table(reports,root/"comparison")
    _csv(root/"comparison"/"endpoints_and_training.csv",rows)
    print(f"Scored {stage} on {partition}: {root/'comparison'}", flush=True)
    if partition=="val" and stage=="matched":
        if any(r["plaque_dice"] is None or not math.isfinite(r["plaque_dice"]) for r in rows):
            raise ValueError("Full-volume plaque Dice must be defined for every arm.")
        winner = min(rows,key=lambda r:(-r["plaque_dice"],r["model"]))
        write_json(out/"matched_selection.json",dict(model=winner["model"],scores=rows,
                   metric="Mean full-volume plaque Dice in positive validation patients; ties by model ID"),lock=True)
    return reports


def paired_table(reports, output):
    import numpy as np
    from plaque_benchmark.evaluation import _csv

    metrics = {}
    for report in reports:
        metrics[report.name] = {(row["case"], row["class"]):row["dice"] for row in read_json(report/"metrics.json")
                                if row["branch"]=="ALL" and row["scope"]=="end_to_end"}
    base = metrics["ffr_unet"]
    rows = []
    for model, values in metrics.items():
        if model == "ffr_unet":
            continue
        for label in ("plaque", "lumen"):
            pairs = [values[key]-base[key] for key in sorted(base) if key[1]==label
                     and base[key] is not None and values[key] is not None]
            if pairs:
                draws = np.random.default_rng(2028).choice(pairs,(20000,len(pairs)),replace=True).mean(axis=1)
                low, high = np.quantile(draws,[.025,.975])
                rows.append(dict(model=model, reference="ffr_unet", endpoint=label, patients=len(pairs),
                                 mean_dice_difference=float(np.mean(pairs)), ci95_low=low, ci95_high=high))
    _csv(output/"paired_dice_vs_ffr.csv",rows)


def historical_table(bench, current, output):
    from plaque_benchmark.evaluation import _csv

    baseline = read_json(current[0]/"summary.json")
    sources = [("matched_architecture_comparison",p) for p in current]
    for name in ("PlaquePilot_away80_tuned_ctta25","PlaquePilot_nnfix80_ctta25","PlaquePilot_CTTAoff_paired80_baseline25"):
        sources.extend((name,p.parent) for p in sorted((bench/"runner"/name/"reports").glob("*/summary.json")))
    rows = []
    for group, path in sources:
        s = read_json(path/"summary.json")
        if (s["run"]["split_sha256"] != baseline["run"]["split_sha256"] or s["run"]["partition"]!="val" or
            s["evaluated_ids"] != baseline["evaluated_ids"] or any(s[k]!=baseline[k] for k in
            ("surface_tolerance_mm","lesion_match_min_iou","empty_gt_dice","branch_attribution"))):
            print(f"Excluded incompatible historical evaluation: {path}",flush=True)
            continue
        rows.append(dict(protocol=group, model=path.name, checkpoint_sha256=s["run"]["checkpoint_sha256"],
                         cases=s["cases"],plaque_dice=s["plaque"]["dice"],lumen_dice=s["lumen"]["dice"],
                         healthy_branch_fp_rate=s.get("healthy_branches",{}).get("fp_rate"), source=str(path)))
    _csv(output/"historical_validation_context.csv",rows)


def run(args):
    from filelock import FileLock
    bench = args.bench.resolve()
    dataset = bench/"prepared/dataset.json"
    manifest = read_json(dataset)
    protocol = read_json(ROOT/"plaque_3d_protocol.json")
    out = (args.out or bench/"runner"/"PlaquePilot_3D_matched120_long240").resolve()
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out/"launcher.lock"), timeout=0):
        audit(bench,out,manifest,protocol,"cuda")
        execute(args,bench,manifest,protocol,out,"cuda")


def execute(args,bench,manifest,protocol,out,device):
    phase=args.phase
    if phase=="legacy-val":
        from plaque_benchmark.round2_confirm import confirm
        args.phase,args.partition,args.decision="confirm","val","Replay archived checkpoint rules on validation."
        args.out=out/"historical_confirmation"
        args.bench,args.dataset,args.device=bench,bench/"prepared/dataset.json",device
        return confirm(args,manifest)
    steps = {"all":("init","tune","train","score","extend"),
             "run":("tune","train","score","extend")}.get(phase,(phase,))
    for step in steps:
        print(f"Starting {step}",flush=True)
        if step=="check":
            continue
        if step=="init":
            init_all(manifest,protocol,out,device)
        elif step=="tune":
            tune(manifest,protocol,out,device)
        elif step=="train":
            train_stage(manifest,protocol,out,device,MODELS,"matched")
        elif step=="score":
            reports = score(manifest,out,device,"val","matched",MODELS)
            historical_table(bench,reports,out/"scoring/val/matched/comparison")
        elif step=="extend":
            selection=read_json(out/"matched_selection.json")
            models=list(MODELS) if args.extend_all else [selection["model"]]
            write_json(out/"promotion.json",dict(models=models, matched_winner=selection["model"]),lock=True)
            train_stage(manifest,protocol,out,device,models,"long")
            score(manifest,out,device,"val","long",models)
        elif step in ("freeze","test"):
            from plaque_benchmark.protection import freeze
            promotion=read_json(out/"promotion.json")
            groups={"matched":list(MODELS),"long":promotion["models"]}
            for name in promotion["models"]:
                path = out/"scoring"/"val"/"long"/"reports"/name/"summary.json"
                if read_json(path)["run"]["checkpoint_sha256"] != file_sha256(out/"long"/name/"best.pt"):
                    raise ValueError(f"Extended checkpoint and validation summary disagree for {name}.")
            path=out/"frozen_test.json"
            checkpoints=[verify_receipt(out/stage/model) for stage,models in groups.items() for model in models]
            test_hashes = case_hashes(select(manifest,"test"))
            if step=="freeze":
                write_json(out/"heldout_input_hashes.json",test_hashes,lock=True)
                freeze(manifest,path,checkpoints=checkpoints,decision="Fixed-budget comparison of three architectures, followed by separate extended arm(s) selected on validation.")
            else:
                if test_hashes != read_json(out/"heldout_input_hashes.json"):
                    raise ValueError("Held-out inputs changed after the freeze.")
                frozen=read_json(path)
                if frozen["candidate_ids"]!=sorted({file_sha256(c) for c in checkpoints}):
                    raise ValueError("Checkpoint set changed after test freeze.")
                for stage,models in groups.items():
                    score(manifest,out,device,"test",stage,models,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=PHASES)
    p.add_argument("--bench",type=Path,default=ROOT/"experiments"/"PlaquePilot_confirmed")
    p.add_argument("--out",type=Path)
    p.add_argument("--extend-all",action="store_true",help="Extend all tuned networks after the fixed comparison.")
    p.add_argument("--nnunet-results",type=Path,help="Historical confirmation only")
    p.add_argument("--nnunet-python",help="Historical confirmation only")
    for name in ("nnunet","nnunet-resenc","ffr-unet"):
        p.add_argument("--"+name,type=Path,help="Historical checkpoint override")
    args=p.parse_args()
    if args.extend_all and args.phase not in ("all","run","extend"):
        p.error("--extend-all applies to all, run, or extend.")
    run(args)


if __name__=="__main__":
    main()

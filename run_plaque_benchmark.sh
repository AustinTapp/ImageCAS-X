#!/usr/bin/env bash
# Save beside pyproject.toml in the extracted ImageCAS-X directory.
# Uses the active native Python, including Windows Conda Python from Git Bash.
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
python_executable="${PLAQUE_PYTHON:-python}"
if command -v cygpath >/dev/null 2>&1; then
    launcher="$(cygpath -w "$launcher")"
    if [[ "$python_executable" == [A-Za-z]:* ]]; then
        python_executable="$(cygpath -u "$python_executable")"
    fi
fi
exec "$python_executable" - "$launcher" "$@" <<'PYTHON_RUNNER'
"""One-file SHK benchmark launcher. Python workers run through package entry points."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

NATIVE = ("cas_net", "ffr_unet", "swin_unetr", "dynunet")
EXTERNAL = {"nnunet": "nnUNetPlans_20G", "nnunet_resenc": "nnUNetResEncUNetLPlans_20G"}
DEFAULT_MODELS = ("cas_net", "ffr_unet", "swin_unetr", "nnunet", "nnunet_resenc")
SCHEMA = 1

PROBE = r'''
import json, os, platform, sys
import torch
from importlib.metadata import version, PackageNotFoundError
value = {"python": platform.python_version(), "executable": sys.executable,
         "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
for package in ("nibabel", "SimpleITK", "monai", "ctaug"):
    try:
        value[package] = version(package)
    except PackageNotFoundError:
        value[package] = None
if value.get('ctaug'):
    import ctaug, hashlib
    from pathlib import Path
    files=sorted(Path(ctaug.__file__).parent.rglob('*.py'))
    value['ctaug_source_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
if sys.argv[1] == "nnunet":
    from importlib.metadata import distribution
    import nnunetv2
    value.update(version=distribution("nnunetv2").version, package=str(next(iter(nnunetv2.__path__))),
                 paths={key: os.environ.get(key) for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")})
print("RUNNER_JSON=" + json.dumps(value))
'''

ENTRY_POINT = r'''
from importlib.metadata import distribution
import sys
name, *arguments = sys.argv[1:]
entry = next((ep for ep in distribution("nnunetv2").entry_points if ep.group == "console_scripts" and ep.name == name), None)
if entry is None:
    raise RuntimeError("Installed nnU-Net has no console entry point " + name)
sys.argv = [name, *arguments]
entry.load()()
'''

VERIFY = r'''
import json, math, sys
from pathlib import Path
import torch
kind, filename, expected_epoch, expected_steps, strict = sys.argv[1:]
path = Path(filename)
if kind == "nnunet" and path.suffix == ".safetensors":
    from nnunetv2.utilities.checkpoints import load_checkpoint_file
    state = load_checkpoint_file(str(path), map_location=torch.device("cpu"))
else:
    # These files are generated locally by this runner's training jobs.
    state = torch.load(str(path), map_location="cpu", weights_only=False)
if kind == "native":
    epoch = int(state["epoch"])
    last = state["history"][-1]
    losses = [float(last["train_loss"]), float(last["val_patch_loss"])]
    steps = int(last["optimizer_steps"])
    weights = state["model"]
    if steps != epoch * int(expected_steps):
        raise ValueError("Native checkpoint optimizer-step count differs from the configured budget.")
else:
    epoch = int(state["current_epoch"])
    logs = state["logging"]
    losses = [float(logs["train_losses"][-1]), float(logs["val_losses"][-1])]
    if min(len(logs["train_losses"]), len(logs["val_losses"])) < epoch:
        raise ValueError("nnU-Net checkpoint lacks the completed epoch's training/validation history.")
    steps = epoch * int(expected_steps)
    weights = state["network_weights"]
if epoch < 1 or (strict == "1" and epoch != int(expected_epoch)):
    raise ValueError("Training did not finish the required epoch count.")
if not all(math.isfinite(loss) for loss in losses) or not weights:
    raise ValueError("Checkpoint has nonfinite loss or missing network weights.")
for name, tensor in weights.items():
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("Checkpoint contains nonfinite weights: " + name)
metrics = {"epoch": epoch, "optimizer_steps": steps,
           "train_loss": losses[0], "val_loss": losses[1]}
if kind == "native":
    metrics.update(peak_vram_gib=last.get("peak_vram_gib"), epoch_seconds=last.get("seconds"))
    if last.get("seconds"):
        previous = state["history"][-2]["patches"] if len(state["history"]) > 1 else 0
        metrics["patches_per_second"] = (last["patches"] - previous) / last["seconds"]
else:
    memory = path.parent / "gpu_memory.json"
    if memory.is_file():
        metrics.update(json.loads(memory.read_text(encoding="utf-8")))
print("RUNNER_JSON=" + json.dumps(metrics))
'''


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint(folder, stem):
    # New nnU-Net uses safetensors, while existing installations commonly use pth.
    for suffix in (".safetensors", ".pth"):
        candidate = Path(folder) / (stem + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No {stem} checkpoint in {folder}")


class Runner:
    def __init__(self, args, launcher):
        self.args = args
        self.launcher = Path(launcher).resolve()
        self.repo = Path(args.repo).resolve() if args.repo else self.launcher.parent
        if not (self.repo / "plaque_benchmark" / "__main__.py").is_file():
            raise FileNotFoundError("Save this .sh beside pyproject.toml in ImageCAS-X, or supply --repo.")
        self.bench = self.resolve(args.bench)
        self.dataset = self.resolve(args.dataset) if args.dataset else self.bench / "prepared" / "dataset.json"
        self.work = self.bench / "runner" / args.run_name
        self.work.mkdir(parents=True, exist_ok=True)
        self.log_path = self.work / f"{args.mode}.log"
        self.records = []
        self.env = dict(os.environ)
        self.env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        self.env.setdefault("nnUNet_compile", "false")
        self.env.setdefault("nnUNet_n_proc_DA", "6" if args.workstation else "8")
        self.env.setdefault("OMP_NUM_THREADS", "1")
        self.env.setdefault("MKL_NUM_THREADS", "1")
        self.env["PYTHONPATH"] = str(self.repo) + os.pathsep + self.env.get("PYTHONPATH", "")
        self.native_python = sys.executable
        self.nn_python = args.nnunet_python or self.env.get("NNUNET_PYTHON") or self.native_python
        self.probes = {}
        self.emit(f"\n{datetime.now(timezone.utc).isoformat()}  mode={args.mode}  models={','.join(args.models)}")

    def resolve(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.repo / path).resolve()

    def emit(self, message, model=None):
        print(message, flush=True)
        for path in (self.log_path, self.work / "logs" / f"{self.args.mode}_{model}.log" if model else None):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")

    def run(self, command, model=None, json_result=False):
        display = [str(value) for value in command]
        if len(display) > 2 and display[1] == "-c":
            label = "environment probe" if command[2] == PROBE else "checkpoint verification"
            if command[2] == ENTRY_POINT:
                label = "installed nnU-Net console entry point"
            display[2] = "<" + label + ">"
        self.emit("COMMAND " + subprocess.list2cmdline(display), model)
        environment = dict(self.env)
        # A separate Windows Conda interpreter also needs its own DLL search paths.
        executable = Path(str(command[0]))
        prefix = executable.parent
        if os.name == "nt" and (prefix / "conda-meta").is_dir():
            directories = [prefix, prefix / "Scripts", prefix / "Library" / "bin",
                           prefix / "Library" / "usr" / "bin", prefix / "Library" / "mingw-w64" / "bin"]
            environment["PATH"] = os.pathsep.join(str(path) for path in directories) + os.pathsep + environment.get("PATH", "")
            environment["CONDA_PREFIX"] = str(prefix)
        process = subprocess.Popen([str(v) for v in command], cwd=self.repo, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        result = None
        try:
            for line in process.stdout:
                line = line.rstrip("\r\n")
                self.emit(line, model)
                if line.startswith("RUNNER_JSON="):
                    result = json.loads(line.removeprefix("RUNNER_JSON="))
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if status:
            raise RuntimeError(f"Command exited with code {status}. See {self.log_path}")
        if json_result and result is None:
            raise RuntimeError("The verification worker returned no result.")
        return result

    def package(self, *arguments, model=None, external=False):
        python = self.nn_python if external else self.native_python
        return self.run([python, "-m", "plaque_benchmark", *arguments], model)

    def nn_command(self, name, *arguments, model=None):
        return self.run([self.nn_python, "-c", ENTRY_POINT, name, *arguments], model)

    def probe(self, external=False):
        kind = "nnunet" if external else "native"
        if kind not in self.probes:
            python = self.nn_python if external else self.native_python
            value = self.run([python, "-c", PROBE, kind], json_result=True)
            if self.args.device == "cuda" and not value["cuda_available"]:
                raise RuntimeError(f"CUDA unavailable in {value['executable']}. Check its PyTorch installation.")
            if external and not all(value["paths"].values()):
                raise RuntimeError("Set nnUNet_raw, nnUNet_preprocessed and nnUNet_results in the external environment.")
            self.probes[kind] = value
        return self.probes[kind]

    def manifest(self):
        data = read(self.dataset)
        for partition in ("train", "val"):
            if not any(row["partition"] == partition for row in data["cases"].values()):
                raise ValueError(f"Prepared dataset has no {partition} patients.")
        if any(row.get("geometry") for row in data["cases"].values()) or data.get("geometry", "native") != "native":
            raise ValueError("This launcher compares native-volume multiclass models. Use the documented sMPR workflow separately.")
        return data

    def native_config(self, model):
        from plaque_benchmark.native import config_for, _validate_settings
        override = None
        if self.args.config_dir:
            candidate = self.resolve(self.args.config_dir) / f"{model}.json"
            override = str(candidate) if candidate.is_file() else None
        value = config_for(model, "screen", None)
        if self.args.a5000:
            # Same effective batch (32) across native models, without changing context.
            batch = {"cas_net": 32, "ffr_unet": 8, "swin_unetr": 4, "dynunet": 8}[model]
            value.update(batch_size=batch, accumulate=32 // batch, max_vram_gib=22.,
                         cache_cases=24, workers=2)
        if self.args.workstation:
            value.update(workers=4, cache_build_workers=4, prefetch_factor=2)
        if override:
            value.update(read(override))
        if self.args.ctta_config:
            from plaque_benchmark.ctta import settings
            value['augmentation']=settings(read(self.resolve(self.args.ctta_config)))
        if self.args.epochs is not None:
            value["epochs"] = self.args.epochs
        if self.args.steps is not None:
            value["steps"] = self.args.steps
        if self.args.workers is not None:
            value["workers"] = self.args.workers
        if self.args.patch:
            value["patch"] = self.args.patch
        _validate_settings(value, "multiclass")
        return value

    @property
    def dataset_name(self):
        return f"Dataset{self.args.nnunet_id:03d}_SHKPlaque"

    def nn_paths(self):
        probe = self.probe(True)
        return {key: Path(value) for key, value in probe["paths"].items()}

    def nn_configuration(self, model):
        paths = self.nn_paths()
        base = paths["nnUNet_preprocessed"] / self.dataset_name
        plan_path = base / f"{EXTERNAL[model]}.json"
        plans = read(plan_path)
        split = read(base / "splits_final.json")
        manifest = self.manifest()
        expected = [{part: sorted(key for key, row in manifest["cases"].items() if row["partition"] == part)
                     for part in ("train", "val")}]
        if split != expected:
            raise ValueError("nnU-Net split differs from the locked SHK split. Run prepare-nnunet.")
        exported = paths["nnUNet_raw"] / self.dataset_name / "benchmark_export.json"
        if read(exported)["dataset_sha256"] != fingerprint(manifest):
            raise ValueError("nnU-Net export belongs to a different manifest. Run prepare-nnunet.")
        config = plans["configurations"]["3d_fullres"]
        data_dir = base / config["data_identifier"]
        cases = expected[0]["train"] + expected[0]["val"]
        for case in cases:
            if not any(path.suffix in {".npz", ".npy", ".b2nd"} for path in data_dir.glob(case + ".*")):
                raise FileNotFoundError(f"Missing preprocessed nnU-Net data for {case} in {data_dir}")
        augmentation = None
        if self.args.ctta_config:
            from plaque_benchmark.ctta import settings
            if config['normalization_schemes'] != ['CTNormalization']:
                raise ValueError('CTTA external adapter requires one CTNormalization channel')
            augmentation = settings(read(self.resolve(self.args.ctta_config)))
        stats=plans['foreground_intensity_properties_per_channel']['0']
        return {"augmentation":augmentation,"spacing":config['spacing'],
                "ct_mean":stats['mean'],"ct_std":stats['std'],
                "augmentation_workers":self.env.get('nnUNet_n_proc_DA'),"plan": EXTERNAL[model], "plans_sha256": sha_file(plan_path),
                "epochs": self.args.epochs or 25, "steps": self.args.steps or 100,
                "val_steps": 48, "batch_size": self.args.nnunet_batch_size or config["batch_size"], "patch": config["patch_size"],
                "reader": plans["image_reader_writer"], "device": self.args.device}

    def context(self, model):
        environment = self.probe(model in EXTERNAL)
        configuration = self.nn_configuration(model) if model in EXTERNAL else self.native_config(model)
        if configuration.get('augmentation') and not environment.get('ctaug'):
            raise ValueError('CTTA requested but ctaug is absent in the selected Python environment. Install requirements_workstation.txt there first.')
        source = {}
        for folder in (self.repo / "plaque_benchmark", self.repo / "models", self.repo / "utils"):
            source.update({str(path.relative_to(self.repo)): sha_file(path) for path in sorted(folder.rglob("*.py"))})
        value = {"schema": SCHEMA, "model": model, "dataset_sha256": sha_file(self.dataset),
                 "configuration": configuration, "environment": environment, "source": source,
                 "launcher_sha256": sha_file(self.launcher), "device": self.args.device,
                 "init_steps": self.args.init_steps, "init_val_patches": self.args.init_val_patches}
        return value, fingerprint(value)

    def verify(self, model, path, steps, expected_epoch=1, strict=True):
        python = self.nn_python if model in EXTERNAL else self.native_python
        return self.run([python, "-c", VERIFY, "nnunet" if model in EXTERNAL else "native", str(path),
                         str(expected_epoch), str(steps), "1" if strict else "0"], model, json_result=True)

    def receipt(self, model):
        return self.work / "init" / model / "PASS.json"

    def gate(self, model, signature):
        receipt = read(self.receipt(model))
        if receipt["signature"] != signature:
            raise ValueError(f"{model} initialization is stale. Run init with the same settings.")
        if receipt["checkpoint_sha256"] != sha_file(receipt["checkpoint"]):
            raise ValueError(f"{model} initialization checkpoint changed or is incomplete.")
        self.verify(model, receipt["checkpoint"], self.args.init_steps)
        if receipt.get("trainer_file") and sha_file(receipt["trainer_file"]) != receipt["trainer_sha256"]:
            raise ValueError("Installed initialization trainer changed. Run init again.")
        return receipt

    def install_trainer(self, model, context, token, initialization):
        nn = self.probe(True)
        configuration = context["configuration"]
        name = "nnUNetTrainerSHKRunner" + token
        epochs = 1 if initialization else configuration["epochs"]
        steps = self.args.init_steps if initialization else configuration["steps"]
        val_steps = self.args.init_val_patches if initialization else configuration["val_steps"]
        text = f'''# Generated by run_plaque_benchmark.sh. Local SHK experiment trainer.
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class {name}(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = {epochs}
        self.num_iterations_per_epoch = {steps}
        self.num_val_iterations_per_epoch = {val_steps}
        self.save_every = 1
        self.configuration_manager.configuration["batch_size"] = {configuration['batch_size']}
        if self.device.type == "cuda":
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., 22 * 2**30 / total), self.device)

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        wrapped = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        augmentation = {configuration.get('augmentation')!r}
        if augmentation:
            from plaque_benchmark.ctta import NNUNetCTTA
            return NNUNetCTTA(wrapped, augmentation, {configuration.get('spacing')!r},
                              {configuration.get('ct_mean')!r}, {configuration.get('ct_std')!r})
        return wrapped

    def on_epoch_start(self):
        import time
        self._memory_epoch_start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return super().on_epoch_start()

    def on_epoch_end(self):
        import json, time
        from pathlib import Path
        result = super().on_epoch_end()
        if self.device.type == "cuda":
            row = dict(peak_vram_gib=torch.cuda.max_memory_reserved(self.device) / 2**30,
                       peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 2**30,
                       epoch_seconds=time.perf_counter() - self._memory_epoch_start,
                       batch_size=int(self.batch_size))
            self.print_to_log_file("GPU_METRICS", row)
            Path(self.output_folder, "gpu_memory.json").write_text(json.dumps(row), encoding="utf-8")
        return result

    def perform_actual_validation(self, *args, **kwargs):
        self.print_to_log_file("SHK runner performs full-volume evaluation from checkpoint_best after training.")
'''
        directory = Path(nn["package"]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
        if not directory.is_dir():
            raise FileNotFoundError(f"External trainer search directory is absent: {directory}")
        path = directory / (name + ".py")
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Conflicting generated trainer: {path}")
        path.write_text(text, encoding="utf-8")
        output = self.nn_paths()["nnUNet_results"] / self.dataset_name / f"{name}__{configuration['plan']}__3d_fullres" / "fold_0"
        return name, path, output

    def audit(self):
        command = ["audit", "--images", self.args.images, "--masks", self.args.masks, "--out", str(self.bench / "audit.json")]
        if self.args.review:
            command += ["--review", str(self.resolve(self.args.review))]
        self.package(*command)
        counts = {}
        for case in read(self.bench / "audit.json")["cases"]:
            counts[case["status"]] = counts.get(case["status"], 0) + 1
        self.emit("Audit status counts " + json.dumps(counts))
        self.emit("Review audit.json and audit.csv before prepare.")

    def prepare(self):
        command = ["split", "--audit", str(self.bench / "audit.json"), "--out", str(self.bench / "split.json")]
        if self.args.require_reviewed:
            command.append("--require-reviewed")
        self.package(*command)
        self.package("prepare", "--audit", str(self.bench / "audit.json"), "--split", str(self.bench / "split.json"),
                     "--out", str(self.bench / "prepared"))
        self.emit(f"Prepared {self.dataset}")

    def prepare_nnunet(self):
        paths = self.nn_paths()
        external_models = [name for name in self.args.models if name in EXTERNAL]
        if not external_models:
            raise ValueError("Select nnunet and/or nnunet_resenc for prepare-nnunet.")
        self.package("nnunet-export", "--dataset", str(self.dataset), "--raw", str(paths["nnUNet_raw"]),
                     "--name", self.dataset_name, external=True)
        for model in external_models:
            planner = "ExperimentPlanner" if model == "nnunet" else "nnUNetPlannerResEncL"
            self.nn_command("nnUNetv2_plan_and_preprocess", "-d", str(self.args.nnunet_id), "-pl", planner,
                "--verify_dataset_integrity", "-np", str(self.args.preprocess_workers), "-npfp", str(self.args.preprocess_workers),
                "-c", "3d_fullres", "-gpu_memory_target", "20", "-overwrite_plans_name", EXTERNAL[model], model=model)
        self.package("nnunet-split", "--exported", str(paths["nnUNet_raw"] / self.dataset_name),
                     "--preprocessed", str(paths["nnUNet_preprocessed"]), external=True)

    def init_model(self, model):
        context, signature = self.context(model)
        passed = self.receipt(model)
        if passed.exists():
            try:
                old = self.gate(model, signature)
                self.emit(f"REUSED successful one-epoch check for {model}", model)
                return {"model": model, "status": "PASS_REUSED", **old}
            except (ValueError, FileNotFoundError):
                self.emit(f"Rechecking changed initialization for {model}", model)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        attempt = self.work / "init" / model / stamp
        attempt.mkdir(parents=True)
        write(attempt / "context.json", context)
        if model in EXTERNAL:
            trainer, trainer_file, output = self.install_trainer(model, context, "Init" + signature[:10] + stamp, True)
            self.nn_command("nnUNetv2_train", str(self.args.nnunet_id), "3d_fullres", "0", "-tr", trainer,
                            "-p", context["configuration"]["plan"], "-device", self.args.device, model=model)
            saved = checkpoint(output, "checkpoint_final")
        else:
            configuration = {**context["configuration"], "epochs": 1, "steps": self.args.init_steps,
                             "validation_patches": self.args.init_val_patches, "patience": 0}
            config_file = attempt / "override.json"
            write(config_file, configuration)
            output = attempt / "training"
            self.package("train", "--dataset", str(self.dataset), "--model", model, "--profile", "screen", "--task", "multiclass",
                         "--config", str(config_file), "--out", str(output), "--device", self.args.device, model=model)
            saved = output / "last.pt"
            if not (output / "best.pt").is_file():
                raise FileNotFoundError("Initialization produced no best checkpoint.")
        metrics = self.verify(model, saved, self.args.init_steps)
        receipt = {"signature": signature, "checkpoint": str(saved), "checkpoint_sha256": sha_file(saved),
                   "metrics": metrics, "completed_utc": datetime.now(timezone.utc).isoformat(), "context": str(attempt / "context.json")}
        if model in EXTERNAL:
            receipt.update(trainer_file=str(trainer_file), trainer_sha256=sha_file(trainer_file))
        write(passed, receipt)
        return {"model": model, "status": "PASS", **receipt}

    def train_model(self, model, context, signature):
        folder = self.work / "training" / model
        folder.mkdir(parents=True, exist_ok=True)
        specification = folder / "specification.json"
        if specification.exists() and read(specification)["signature"] != signature:
            old = read(specification)
            def comparable(value):
                value = json.loads(json.dumps(value))
                value.pop("launcher_sha256", None)
                value["source"] = {k: v for k, v in value["source"].items()
                                   if k.replace("\\", "/") != "plaque_benchmark/native.py"}
                return value
            if not (self.args.adopt_performance_update and self.args.resume and model in NATIVE
                    and comparable(old["context"]) == comparable(context)):
                raise ValueError("Training settings changed. For a native.py-only performance update, rerun init then use --resume --adopt-performance-update. Otherwise select a new run name.")
            backup = folder / ("specification.before_performance_" + old["signature"][:16] + ".json")
            if not backup.exists():
                write(backup, old)
            self.emit(f"Adopting native.py performance update for {model}; training configuration and other source files match.", model)
        write(specification, {"signature": signature, "context": context})
        import time
        start = time.perf_counter()
        if model in EXTERNAL:
            trainer, trainer_file, output = self.install_trainer(model, context, "Screen" + signature[:18], False)
            if output.exists() and any(output.iterdir()) and not self.args.resume:
                raise FileExistsError(f"Use --resume for existing training output {output}")
            options = []
            if self.args.resume and any(output.glob("checkpoint_*")):
                options = ["--c"]
            self.nn_command("nnUNetv2_train", str(self.args.nnunet_id), "3d_fullres", "0", "-tr", trainer,
                            "-p", context["configuration"]["plan"], "-device", self.args.device, *options, model=model)
            saved = checkpoint(output, "checkpoint_final")
            best = checkpoint(output, "checkpoint_best")
        else:
            config = folder / "override.json"
            write(config, context["configuration"])
            output = folder / "model"
            options = ["--resume"] if self.args.resume and (output / "last.pt").is_file() else []
            self.package("train", "--dataset", str(self.dataset), "--model", model, "--profile", "screen", "--task", "multiclass",
                         "--config", str(config), "--out", str(output), "--device", self.args.device, *options, model=model)
            saved, best = output / "last.pt", output / "best.pt"
        hours = (time.perf_counter() - start) / 3600
        metrics = self.verify(model, saved, context["configuration"]["steps"],
                              expected_epoch=context["configuration"]["epochs"], strict=model in EXTERNAL)
        best_id = sha_file(best)
        predictions = self.work / "predictions" / model / best_id[:16]
        report = self.work / "reports" / model
        if model in EXTERNAL:
            if context["configuration"]["reader"] != "SimpleITKIO":
                raise ValueError("Automatic nnU-Net import requires SimpleITKIO plans. Inspect probability axes for this plan.")
            inputs = self.work / "nnunet_val_inputs"
            self.package("nnunet-inputs", "--dataset", str(self.dataset), "--partition", "val", "--out", str(inputs))
            scores = folder / ("nnunet_scores_" + best_id[:16])
            self.nn_command("nnUNetv2_predict", "-d", str(self.args.nnunet_id), "-i", str(inputs), "-o", str(scores),
                "-c", "3d_fullres", "-f", "0", "-tr", trainer, "-p", context["configuration"]["plan"],
                "-chk", best.name, "--save_probabilities", "-device", self.args.device, model=model)
            self.package("nnunet-import", "--dataset", str(self.dataset), "--partition", "val", "--predictions", str(scores),
                "--out", str(predictions), "--task", "multiclass", "--checkpoint", str(best),
                "--initialization", "random", external=True, model=model)
        else:
            self.package("infer", "--dataset", str(self.dataset), "--partition", "val", "--checkpoint", str(best),
                         "--out", str(predictions), "--device", self.args.device, model=model)
        self.package("evaluate", "--dataset", str(self.dataset), "--partition", "val", "--predictions", str(predictions),
                     "--out", str(report), model=model)
        return {"model": model, "status": "PASS", "checkpoint": str(best), "report": str(report),
                "training_wall_hours_this_invocation": hours, **metrics}

    def summary(self):
        write(self.work / f"{self.args.mode}_summary.json", self.records)
        fields = ["model", "status", "epoch", "optimizer_steps", "train_loss", "val_loss", "peak_vram_gib", "epoch_seconds", "patches_per_second", "log", "error", "report"]
        with (self.work / f"{self.args.mode}_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.records:
                writer.writerow({**row, **row.get("metrics", {})})

    def execute(self):
        if self.args.mode in {"audit", "prepare", "prepare-nnunet"}:
            getattr(self, self.args.mode.replace("-", "_"))()
            return
        self.manifest()
        contexts = {}
        if self.args.mode == "train":
            # Verify every selected model before starting the first long training job.
            for model in self.args.models:
                context, signature = self.context(model)
                self.gate(model, signature)
                contexts[model] = context, signature
        for model in self.args.models:
            self.emit(f"\nSTART {self.args.mode} {model}", model)
            try:
                row = (self.init_model(model) if self.args.mode == "init"
                       else self.train_model(model, *contexts[model]))
                row["log"] = str(self.work / "logs" / f"{self.args.mode}_{model}.log")
                self.records.append(row)
                self.emit(f"PASS {self.args.mode} {model}", model)
            except Exception as exc:
                self.emit(traceback.format_exc(), model)
                self.records.append({"model": model, "status": "FAIL", "error": str(exc),
                    "log": str(self.work / "logs" / f"{self.args.mode}_{model}.log")})
                self.summary()
                raise
            self.summary()
        if self.args.mode == "train":
            reports = [row["report"] for row in self.records]
            self.package("compare", "--reports", *reports, "--out", str(self.work / "comparison"))
        else:
            self.emit("Initialization checks complete. Review init_summary.csv and per-model logs. Start train in a separate command.")


def parse(argv):
    parser = argparse.ArgumentParser(description="Sequential SHK training with a verified one-epoch gate.")
    parser.add_argument("mode", choices=("audit", "prepare", "prepare-nnunet", "init", "train"))
    parser.add_argument("--repo", help="ImageCAS-X root; default is the directory containing this .sh")
    parser.add_argument("--bench", default="experiments/plaque_benchmark")
    parser.add_argument("--dataset", help="Prepared dataset.json; defaults to BENCH/prepared/dataset.json")
    parser.add_argument("--images", default=r"D:\Data\PlaqueDataset\AsNifti")
    parser.add_argument("--masks", default=r"D:\Data\PlaqueDataset\MasksUnverified")
    parser.add_argument("--review", help="Audit decision CSV with case,status columns")
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--models", nargs="+", choices=(*NATIVE, *EXTERNAL), default=list(DEFAULT_MODELS))
    parser.add_argument("--nnunet-python", help="External environment's python executable, or NNUNET_PYTHON")
    parser.add_argument("--nnunet-id", type=int, default=701)
    parser.add_argument("--preprocess-workers", type=int, default=8)
    parser.add_argument("--run-name", default="screen25")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, help="Screening epochs; default 25")
    parser.add_argument("--steps", type=int, help="Screening optimizer steps per epoch; default 100")
    parser.add_argument("--init-steps", type=int, default=100, help="Optimizer steps in the single initialization epoch; default 100")
    parser.add_argument("--init-val-patches", type=int, default=2, help="Initialization validation budget; native trainer visits at least one patch per validation patient")
    parser.add_argument("--workers", type=int, help="Native DataLoader workers for both init and train")
    parser.add_argument("--patch", type=int, nargs=3, help="Native XYZ patch dimensions for both init and train")
    parser.add_argument("--config-dir", help="Optional native model overrides named cas_net.json, ffr_unet.json, etc.")
    parser.add_argument("--a5000", action="store_true", help="Native batch/accumulation: CAS 32/1, FFR 8/4, Swin 4/8; 64-cube screen patches, cache 24 per worker, 22 GiB cap. Reuse flag for init/train.")
    parser.add_argument("--nnunet-batch-size", type=int, help="Override external batch size without changing patch/spacing or preprocessing; verify with init first. Default uses each plan.")
    parser.add_argument("--adopt-performance-update", action="store_true", help="With --resume and fresh init receipts, adopt native.py/launcher-only changes while preserving identical training settings.")
    parser.add_argument("--workstation", action="store_true", help="Native 4 loading/4 cache-build workers; external 6 augmentation workers unless overridden by environment.")
    parser.add_argument("--ctta-config", help="Training-only CTAug JSON for native and external models. Use a new experiment name.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted full training")
    args = parser.parse_args(argv)
    if args.adopt_performance_update and (args.mode != "train" or not args.resume):
        parser.error("--adopt-performance-update requires train --resume and fresh initialization checks.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name must contain letters, digits, underscores or hyphens.")
    if len(args.models) != len(set(args.models)):
        parser.error("Select each model once.")
    for name in ("epochs", "steps", "init_steps", "init_val_patches", "preprocess_workers", "nnunet_id", "nnunet_batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be nonnegative.")
    return args


def main():
    launcher = sys.argv[1]
    args = parse(sys.argv[2:])
    runner = None
    try:
        runner = Runner(args, launcher)
        sys.path.insert(0, str(runner.repo))
        os.chdir(runner.repo)
        runner.execute()
    except KeyboardInterrupt:
        print("Interrupted. Logs and completed checkpoints were retained.", flush=True)
        return 130
    except Exception:
        message = traceback.format_exc()
        if runner is not None:
            runner.emit(message)
            write(runner.work / (args.mode + "_failure.json"), {"error": message, "records": runner.records})
        else:
            print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYTHON_RUNNER

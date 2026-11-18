#!/usr/bin/env python3
"""Collect PlaquePilot results and configurations using only the Python standard library.

Run from ImageCAS-X:
    python combine_plaque_experiment_reports.py --root experiments --out experiments/plaque_report_bundle --zip

Each --root may be an experiment tree or an extracted comparison folder.
Use --no-case-details to omit patient-level, branch-level and lesion tables from the export.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import zipfile
from collections import defaultdict


def read_json(path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flatten(data, prefix=""):
    output = {}
    for name, value in data.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            output.update(flatten(value, key))
        else:
            output[key] = value
    return output


def fingerprint(value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def from_csv(value):
    if value is None or value == "":
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(value)
            return parsed if isinstance(parsed, (bool, int, float, list, dict)) else value
        except (ValueError, TypeError, SyntaxError):
            continue
    return value


def scalar(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    # Keep imported strings inert when the CSV is opened in a spreadsheet.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        try:
            float(value)
        except ValueError:
            return "'" + value
    return value


def study_rows(path):
    """Read the Optuna RDB tables under a read-only SQLite transaction."""
    sql = """
        SELECT s.study_name, t.trial_id, t.number, t.state,
               t.datetime_start, t.datetime_complete, v.value, v.value_type
        FROM trials t JOIN studies s ON t.study_id = s.study_id
        LEFT JOIN trial_values v ON t.trial_id = v.trial_id AND v.objective = 0
        ORDER BY s.study_name, t.number
    """
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        attrs = defaultdict(dict)
        for row in db.execute("SELECT trial_id, key, value_json FROM trial_user_attributes"):
            attrs[row["trial_id"]][row["key"]] = json.loads(row["value_json"])
        params = defaultdict(dict)
        for row in db.execute("SELECT trial_id, param_name, param_value FROM trial_params"):
            # PlaquePilot v1 uses only FloatDistribution search parameters.
            params[row["trial_id"]][row["param_name"]] = row["param_value"]
        rows = []
        for row in db.execute(sql):
            value = dict(row)
            model = value.pop("study_name")
            trial_id = value.pop("trial_id")
            value["full_training_config"] = attrs[trial_id].get("full_training_config")
            value["parameters"] = params[trial_id]
            rows.append((model, value))
    return rows


class Collector:
    def __init__(self, include_cases):
        self.tables = defaultdict(list)
        self.include_cases = include_cases
        self.training = {}
        self.training_directories = {}
        self.evaluations = {}
        self.winner_exports = {}

    def warn(self, path, error):
        self.tables["collection_notes"].append(dict(source=str(path), note=str(error)))

    def load(self, path, reader=read_json):
        if path.is_file():
            try:
                return reader(path)
            except (OSError, ValueError, csv.Error) as exc:
                self.warn(path, exc)
        return None

    def add(self, table, row):
        self.tables[table].append(row)

    def add_index(self, kind, experiment_id, source, **fields):
        self.add("experiment_index", dict(experiment_id=experiment_id, kind=kind,
                                          source=str(source), **fields))

    def config(self, experiment_id, role, config, source):
        if isinstance(config, dict):
            self.add("configurations", dict(experiment_id=experiment_id, role=role,
                     configuration_id=fingerprint(config), source=str(source), config_json=config,
                     **flatten(config, "config")))

    def collect_training(self, paths):
        folders = set()
        standalone = set()
        for path in paths:
            for parent in path.parents:
                if parent.parent.name == "training":
                    folders.add(parent)
                    break
            else:
                if (path.name == "history.csv" and (path.parent / "config.json").is_file()
                    and not any(part == "init" or part.startswith("trial_") for part in path.parts)):
                    standalone.add(path.parent)
        for folder in sorted(folders | standalone):
            specification = self.load(folder / "specification.json") or {}
            context = specification.get("context") or {}
            configuration = context.get("configuration")
            source = folder / "specification.json"
            if not isinstance(configuration, dict):
                for source in (folder / "override.json", folder / "model" / "config.json", folder / "config.json"):
                    configuration = self.load(source)
                    if isinstance(configuration, dict):
                        break
            history = next((p for p in (folder / "model" / "history.csv", folder / "history.csv")
                            if p.is_file()), None)
            if configuration is None and history is None:
                continue
            model = folder.name if folder in folders else (configuration or {}).get("model", folder.name)
            run = folder.parent.parent if folder in folders else folder
            run_id = fingerprint(["training", str(folder)])
            epochs = self.load(history, read_csv) if history else []
            last = epochs[-1] if epochs else {}
            self.training[(run, model)] = run_id
            for directory in (folder, folder / "model"):
                self.training_directories[directory] = run_id
            environment = context.get("environment") or {}
            self.add("training_runs", dict(training_id=run_id, model=model, run=run.name,
                     requested_epochs=(configuration or {}).get("epochs"),
                     requested_steps=(configuration or {}).get("steps"),
                     recorded_epochs=last.get("epoch"), gpu_hours=last.get("gpu_hours_cumulative"),
                     signature=specification.get("signature"), dataset_sha256=context.get("dataset_sha256"),
                     source=str(source if source.is_file() else folder),
                     **flatten(environment, "environment")))
            self.config(run_id, "training", configuration, source)
            for row in epochs or []:
                self.add("training_history", dict(training_id=run_id, **row, source=str(history)))

    def collect_evaluations(self, paths):
        records = []
        for path in paths:
            summary = self.load(path)
            if not isinstance(summary, dict) or not isinstance(summary.get("run"), dict):
                continue
            run = summary["run"]
            if not any(key in summary for key in ("plaque", "lumen", "healthy_branches")):
                continue
            # The launchers make labelled copies for figures. Their numerical summary is identical.
            signature = fingerprint({**summary, "run": {k: v for k, v in run.items() if k != "display_name"}})
            mirror = path.parent.parent.name in {"paired_reports", "comparison_inputs"}
            records.append((mirror, path, summary, signature))
        originals = defaultdict(list)
        for mirror, path, summary, signature in records:
            if not mirror:
                originals[signature].append(path)
        for mirror, path, summary, signature in sorted(records):
            if mirror and len(originals[signature]) == 1:
                self.add("report_aliases", dict(experiment_id=fingerprint(["evaluation", str(originals[signature][0])]),
                         alias=str(path), label=summary["run"].get("display_name")))
                continue
            if mirror and len(originals[signature]) > 1:
                self.warn(path, "Copied report matches multiple originals. Copy kept as a separate source.")
            self.evaluation(path, summary, mirror)

    def evaluation(self, path, summary, mirror):
        meta = summary["run"]
        report = path.parent
        model = meta.get("model")
        run_name = report.parent.name
        training_id = None
        if report.parent.name == "reports":
            model, run_path = report.name, report.parent.parent
            run_name = run_path.name
            training_id = self.training.get((run_path, model))
        if training_id is None and meta.get("checkpoint"):
            training_id = self.training_directories.get(Path(meta["checkpoint"]).resolve().parent)
        case_ids = summary.get("evaluated_ids")
        split, partition = meta.get("split_sha256"), meta.get("partition")
        cohort = fingerprint([split, partition, sorted(case_ids)]) if split and partition and case_ids else None
        plaque, lumen = summary.get("plaque") or {}, summary.get("lumen") or {}
        healthy = summary.get("healthy_branches") or {}
        experiment_id = fingerprint(["evaluation", str(path)])
        self.evaluations[report] = experiment_id
        protocol = {k: summary.get(k) for k in ("surface_tolerance_mm", "lesion_match_min_iou",
                    "lesion_matching", "branch_attribution", "empty_gt_dice", "empty_branch_policy")}
        gpu_hours = meta.get("gpu_hours")
        if gpu_hours is None and training_id:
            gpu_hours = next((r["gpu_hours"] for r in self.tables["training_runs"]
                              if r["training_id"] == training_id), None)
        common = dict(model=model, run=run_name, label=meta.get("display_name") or report.name,
                      objective_kind="full_volume_evaluation", partition=partition, cohort_id=cohort,
                      cases=summary.get("cases"), plaque_dice=plaque.get("dice"),
                      lumen_dice=lumen.get("dice"), healthy_branch_fp_rate=healthy.get("fp_rate"))
        self.add_index("evaluation", experiment_id, path, **common)
        self.add("evaluation_summary", dict(experiment_id=experiment_id, **common,
                 training_id=training_id, copied_report=mirror, task=meta.get("task"), geometry=meta.get("geometry"),
                 initialization=meta.get("initialization"), gpu_hours=gpu_hours, split_sha256=split,
                 protocol_id=fingerprint(protocol), checkpoint=meta.get("checkpoint"),
                 checkpoint_sha256=meta.get("checkpoint_sha256"), candidate_id=meta.get("candidate_id"),
                 plaque_precision=plaque.get("precision"), plaque_recall=plaque.get("recall"),
                 healthy_branches=healthy.get("n"), source=str(path), **protocol,
                 **flatten(summary, "report")))
        self.config(experiment_id, "inference_parameters", meta.get("parameters"), path)
        if meta.get("training_augmentation"):
            self.config(experiment_id, "recorded_training_augmentation", meta["training_augmentation"], path)
        if self.include_cases:
            for table, base in (("evaluation_case_metrics", "metrics"), ("evaluation_lesions", "lesions")):
                data_path = report / (base + ".csv")
                rows = self.load(data_path, read_csv)
                if rows is None:
                    data_path = report / (base + ".json")
                    rows = self.load(data_path)
                for row in rows or []:
                    self.add(table, {**row, "experiment_id": experiment_id, "source": str(data_path)})

    def collect_search(self, folders):
        for folder in sorted(folders):
            trials = {}
            for path in sorted(folder.glob("*_trials.csv")):
                model = path.name.removesuffix("_trials.csv")
                for raw in self.load(path, read_csv) or []:
                    try:
                        trial = int(raw["number"])
                    except (KeyError, ValueError) as exc:
                        self.warn(path, f"Unreadable trial number: {exc}")
                        continue
                    row = {key: from_csv(value) for key, value in raw.items()}
                    row["parameters"] = {k[7:]: v for k, v in row.items() if k.startswith("params_")}
                    row["full_training_config"] = row.get("user_attrs_full_training_config")
                    trials[model, trial] = (path, row)
            db = folder / "studies.sqlite3"
            if db.is_file():
                try:
                    for model, row in study_rows(db):
                        key = (model, row["number"])
                        _, exported = trials.get(key, (None, {}))
                        trials[key] = (db, {**exported, **row})
                except (sqlite3.Error, ValueError) as exc:
                    self.warn(db, f"SQLite inventory unavailable. Used CSV/request records where present. {exc}")
            for request in sorted(folder.glob("*/trial_*/request.json")):
                model = request.parent.parent.name
                try:
                    trial = int(request.parent.name.removeprefix("trial_"))
                except ValueError:
                    continue
                trials.setdefault((model, trial), (request, dict(number=trial, state="UNKNOWN")))
            best = {}
            for best_file in sorted(folder.glob("*_best.json")):
                winner = self.load(best_file)
                if not isinstance(winner, dict) or "trial" not in winner:
                    continue
                model = best_file.name.removesuffix("_best.json")
                best[model] = winner
                export = folder / "best_config" / (model + ".json")
                trial_id = fingerprint(["trial", str(folder), model, winner["trial"]])
                self.winner_exports[export] = trial_id
                self.add("optuna_winners", dict(experiment_id=trial_id, model=model, study=str(folder),
                                              source=str(best_file), **flatten(winner)))
                self.config(trial_id, "exported_winner", self.load(export), export)
                trials.setdefault((model, winner["trial"]), (best_file, dict(
                    number=winner["trial"], state="COMPLETE", value=winner.get("score"),
                    parameters=winner.get("parameters"))))
            for (model, number), (source, row) in sorted(trials.items()):
                request_path = folder / model / f"trial_{number:05d}" / "request.json"
                request = self.load(request_path) or {}
                winner = best.get(model, {})
                trial_id = fingerprint(["trial", str(folder), model, number])
                objective = row.get("value")
                metric = request.get("metric") or winner.get("metric")
                selected = winner.get("trial") == number
                common = dict(model=model, run=folder.name, label=f"{model} trial {number}",
                              status=row.get("state"), objective_kind="validation_patch_screening",
                              objective_value=objective, objective_metric=metric)
                self.add_index("optuna_trial", trial_id, source, **common)
                self.add("optuna_trials", dict(experiment_id=trial_id, **common,
                         study=str(folder), trial=number, exported_winner=selected,
                         study_identity=winner.get("identity"), source=str(source),
                         dataset=request.get("dataset"), device=request.get("device"),
                         **flatten(row, "trial_data")))
                self.config(trial_id, "trial_actual", request.get("config"), request_path)
                self.config(trial_id, "trial_full_training_proposal", row.get("full_training_config"), source)
                history = request_path.parent / "model" / "history.csv"
                for epoch in self.load(history, read_csv) or []:
                    self.add("training_history", dict(experiment_id=trial_id, **epoch, source=str(history)))

    def collect_sweep(self, path):
        folder = path.parent
        winner = self.load(folder / "provisional_winner.json") or {}
        identity = self.load(folder / "validation_identity.json") or {}
        options = {row["name"]: row for row in identity.get("specs", [])}
        meta = self.load(folder / "provisional_predictions" / "run.json") or {}
        split = meta.get("split_sha256")
        case_path = folder / "validation_case_metrics.csv"
        case_rows = self.load(case_path, read_csv) or []
        cases = defaultdict(set)
        for row in case_rows:
            cases[row.get("candidate")].add(row.get("case"))
        for row in self.load(path, read_csv) or []:
            candidate = row.get("candidate")
            experiment_id = fingerprint(["sweep", str(folder), candidate])
            spec = options.get(candidate)
            model = "+".join(spec.get("sources", [])) if spec else None
            cohort = fingerprint([split, "val", sorted(cases[candidate])]) if split and cases[candidate] else None
            common = dict(model=model, run=folder.name, label=candidate, partition="val",
                          cohort_id=cohort, cases=row.get("patients"),
                          objective_kind="validation_full_volume_plaque_sweep",
                          plaque_dice=row.get("plaque_dice"),
                          healthy_branch_fp_rate=row.get("healthy_branch_fp_rate"))
            self.add_index("validation_candidate", experiment_id, path, **common)
            selected = winner.get("spec", {}).get("name") == candidate
            self.add("validation_candidates", {**row, **common, "experiment_id": experiment_id,
                     "split_sha256": split, "provisional_winner": selected, "source": str(path)})
            self.config(experiment_id, "threshold_component_ensemble", spec, folder / "validation_identity.json")
        if self.include_cases:
            for row in case_rows:
                self.add("validation_case_metrics", {**row, "source": str(case_path),
                         "experiment_id": fingerprint(["sweep", str(folder), row.get("candidate")])})

    def collect_events(self, path):
        folder = path.parent
        if path.name == "workflow.json":
            data = self.load(path) or {}
            stages = data.get("stages") or {}
            rows = [dict(stage=name, **row) for name, row in stages.items()]
            for name, search in stages.items():
                if not name.startswith("search/") or search.get("status") != "PASS":
                    continue
                model = name.split("/", 1)[1]
                trained = stages.get(f"tuned-train/{model}", {})
                export_path, report = search.get("winner"), trained.get("report")
                if trained.get("status") == "PASS" and export_path and report:
                    self.add("optuna_promotion_links", dict(model=model,
                             trial_experiment_id=self.winner_exports.get(Path(export_path).resolve()),
                             evaluation_experiment_id=self.evaluations.get(Path(report).resolve()),
                             winner_config=export_path, report=report, source=str(path)))
        elif path.name == "ablation_status.json":
            data = self.load(path) or {}
            rows = [dict(stage="ablation", **row) for row in data.get("failures", [])]
            failed = {row.get("model") for row in rows}
            rows.extend(dict(stage="ablation", model=model, status="PASS", ablation_run=data.get("run"))
                        for model in data.get("models", []) if model not in failed)
        elif path.suffix == ".csv":
            rows = self.load(path, read_csv) or []
        else:
            rows = self.load(path) or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model = row.get("model") or (row.get("record") or {}).get("model")
            training_id = self.training.get((folder, model))
            if path.name in ("init_summary.json", "init_summary.csv") and model:
                context_path = row.get("context")
                if context_path:
                    local = Path(context_path)
                    if not local.is_file():
                        parts = str(context_path).replace("\\", "/").split("/")
                        context_parent = parts[-2] if len(parts) >= 2 else ""
                        local = folder / "init" / model / context_parent / "context.json"
                    context = self.load(local) or {}
                    init_id = fingerprint(["init", str(path), model, context_path])
                    self.config(init_id, "initialization", context.get("configuration"), local)
                    row = {**row, "initialization_id": init_id}
            self.add("execution_status", {"run": folder.name, **flatten(row),
                     "training_id": training_id, "source": str(path)})


def discover(roots, out):
    names = {"summary.json", "specification.json", "override.json", "config.json", "history.csv",
             "comparison.csv", "validation_candidates.csv", "studies.sqlite3", "request.json",
             "model_summary.csv", "workflow.json", "init_summary.json", "init_summary.csv",
             "train_summary.json", "train_summary.csv", "ablation_status.json"}
    paths = set()
    for root in roots:
        for parent, dirs, files in os.walk(root):
            folder = Path(parent)
            dirs[:] = sorted(d for d in dirs if d not in {"predictions", "review_assets", "assets",
                             "nnunet_val_inputs", "case_cache", "__pycache__", ".git",
                             "approved", "rejected", "unreviewed", "needs_review", "nnUNet_raw",
                             "nnUNet_preprocessed", "nnUNet_results"}
                             and not d.startswith(("nnunet_scores_", "."))
                             and (folder / d).resolve() != out)
            for name in files:
                if name in names or name.endswith(("_trials.csv", "_best.json")):
                    paths.add(folder / name)
    return sorted(paths)


TABLES = {
    "experiment_index": "One row per detailed evaluation, validation candidate, or Optuna trial.",
    "evaluation_summary": "Full-volume benchmark summaries, including original metric denominators and protocol.",
    "configurations": "Recorded settings in flattened columns and JSON. Join experiment_id to experiment_index, training_runs.training_id, or execution_status.initialization_id.",
    "training_runs": "Training attempts with recorded configuration or history, including attempts without evaluation.",
    "training_history": "Available native training and search trial epoch records. Join by training_id or trial experiment_id.",
    "optuna_trials": "Trial states, parameter values and recorded validation patch objectives.",
    "optuna_winners": "Exported Optuna selections and their provenance.",
    "optuna_promotion_links": "Workflow links between selected trials and completed tuned full-volume evaluations. IDs are filled when both referenced paths are available in the scan.",
    "validation_candidates": "Plaque probability sweep, component filtering and ensemble outcomes.",
    "execution_status": "Initialization, training and workflow statuses, including recorded errors.",
    "comparison_snapshots": "Aggregate comparison CSVs. These can repeat detailed results or have unknown cohort provenance.",
    "report_aliases": "Labelled copies linked to an unambiguous original evaluation.",
    "evaluation_case_metrics": "Full-volume case and branch metrics, including scope and class.",
    "evaluation_lesions": "Available lesion rows from full-volume evaluation.",
    "validation_case_metrics": "Paired patient-level sweep results.",
    "collection_notes": "Read errors, ambiguous copies and missing report data notes.",
}


def write_bundle(collector, out, zip_output):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "_collection_outputs.json"
    archive_name = out.name + ".zip"
    known_names = {name + ".csv" for name in TABLES} | {"README.md", archive_name}
    previously_generated = set(read_json(manifest)) & known_names if manifest.is_file() else set()
    conflicts = sorted(name for name in known_names - previously_generated if (out / name).exists())
    if conflicts:
        raise ValueError(f"Use a dedicated --out folder. Existing files would conflict: {', '.join(conflicts)}")
    with tempfile.TemporaryDirectory(prefix=".collect-", dir=out) as temp:
        staging = Path(temp)
        header = ["# PlaquePilot experiment report bundle", "",
                  "Use experiment_index.csv to locate evaluations, sweep candidates and search trials.",
                  "Use evaluation_summary.csv for network comparisons and join configurations through training_id.",
                  "Optuna patch objectives and full-volume Dice use different evaluation scopes.",
                  "Match cohort_id, partition, task and evaluation protocol when comparing results.",
                  "A blank cohort_id means the input lacks sufficient information to confirm the same patient set.",
                  "IDs are generated from input paths. A different checkout may produce different IDs.",
                  "Blank metrics are missing or undefined. Aggregate snapshots may duplicate detailed results.",
                  ""]
        for name, description in TABLES.items():
            rows = collector.tables[name]
            if not rows:
                continue
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with (staging / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows({key: scalar(value) for key, value in row.items()} for row in rows)
            header.append(f"- `{name}.csv` ({len(rows)} rows). {description}")
        (staging / "README.md").write_text("\n".join(header) + "\n", encoding="utf-8")
        generated = sorted(staging.iterdir())
        if zip_output:
            with zipfile.ZipFile(staging / archive_name, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in generated:
                    archive.write(path, path.name)
        products = sorted(staging.iterdir())
        product_names = {path.name for path in products}
        # Claim these generated names before publishing so interrupted collections can be retried.
        marker = staging / manifest.name
        marker.write_text(json.dumps(sorted(previously_generated | product_names)), encoding="utf-8")
        os.replace(marker, manifest)
        for path in products:
            os.replace(path, out / path.name)
        for name in previously_generated - product_names:
            (out / name).unlink(missing_ok=True)
        marker.write_text(json.dumps(sorted(product_names)), encoding="utf-8")
        os.replace(marker, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", help="Input tree. Repeat as needed. Defaults to experiments.")
    parser.add_argument("--out", type=Path, default=Path("experiments/plaque_report_bundle"))
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP of generated report tables.")
    parser.add_argument("--no-case-details", action="store_true", help="Omit detailed rows and patient ID lists from exported tables.")
    args = parser.parse_args()
    roots = [root.resolve() for root in (args.root or [Path("experiments")])]
    out = args.out.resolve()
    for root in roots:
        if not root.is_dir():
            parser.error(f"Input folder does not exist: {root}")
        if root == out or out in root.parents:
            parser.error("Choose an output folder distinct from and not above any input root.")
    paths = discover(roots, out)
    collector = Collector(not args.no_case_details)
    collector.collect_training(paths)
    collector.collect_evaluations([path for path in paths if path.name == "summary.json"])
    collector.collect_search({p.parent for p in paths if p.name == "studies.sqlite3" or
                             p.name.endswith(("_trials.csv", "_best.json"))} |
                             {p.parents[2] for p in paths if p.name == "request.json"
                              and p.parent.name.startswith("trial_")})
    for path in paths:
        if path.name == "validation_candidates.csv":
            collector.collect_sweep(path)
        elif path.name == "comparison.csv":
            for row in collector.load(path, read_csv) or []:
                collector.add("comparison_snapshots", {**row, "comparison": path.parent.name, "source": str(path)})
        elif path.name in {"model_summary.csv", "workflow.json", "init_summary.json", "init_summary.csv",
                           "train_summary.json", "train_summary.csv", "ablation_status.json"}:
            if path.suffix != ".csv" or not path.with_suffix(".json").is_file() or path.name == "model_summary.csv":
                try:
                    collector.collect_events(path)
                except (TypeError, AttributeError, ValueError) as exc:
                    collector.warn(path, exc)
    if not any(collector.tables.values()):
        parser.error("No PlaquePilot reports, trial records or configurations found under the input folders.")
    if args.no_case_details:
        for row in collector.tables["evaluation_summary"]:
            for key in ("report.evaluated_ids", "report.excluded_calibration_ids",
                        "report.unevaluable_empty_reference_branches"):
                row.pop(key, None)
    try:
        write_bundle(collector, out, args.zip)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cannot write report bundle: {exc}\n")
    print(f"Report bundle: {out}")
    for table, rows in collector.tables.items():
        if rows:
            print(f"  {table}: {len(rows)}")
    if args.zip:
        print(f"Shareable tables archive: {out / (out.name + '.zip')}")


if __name__ == "__main__":
    main()

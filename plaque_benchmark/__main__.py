"""Independent stages for the SHK lumen/plaque benchmark."""
import argparse
from pathlib import Path
from .data import TASKS, audit, file_sha256, make_split, prepare, read_json, write_json
from .protection import guard

DEFAULT = "experiments/plaque_benchmark"


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="stage",required=True)

    def stage(name, dataset=False, partition=False):
        p = commands.add_parser(name)
        if dataset:
            p.add_argument("--dataset",default=f"{DEFAULT}/prepared/dataset.json")
        if partition:
            p.add_argument("--partition",choices=("train","val","test"),default="val")
            p.add_argument("--frozen")
        return p

    p = stage("synthetic")
    p.add_argument("--out",required=True)
    p.add_argument("--cases",type=int,default=9)
    p = stage("audit")
    p.add_argument("--images",default=r"D:\Data\PlaqueDataset\AsNifti")
    p.add_argument("--masks",default=r"D:\Data\PlaqueDataset\MasksUnverified")
    p.add_argument("--review")
    p.add_argument("--out",default=f"{DEFAULT}/audit.json")
    p = stage("split")
    p.add_argument("--audit",default=f"{DEFAULT}/audit.json")
    p.add_argument("--out",default=f"{DEFAULT}/split.json")
    p.add_argument("--seed",type=int,default=2028)
    p.add_argument("--require-reviewed",action="store_true")
    p = stage("prepare")
    p.add_argument("--audit",default=f"{DEFAULT}/audit.json")
    p.add_argument("--split",default=f"{DEFAULT}/split.json")
    p.add_argument("--out",default=f"{DEFAULT}/prepared")
    p = stage("train",dataset=True)
    p.add_argument("--model",choices=("cas_net","ffr_unet","swin_unetr","dynunet","swin_unetr_v2","mednext"),required=True)
    p.add_argument("--task",choices=tuple(TASKS),default="multiclass")
    p.add_argument("--profile",choices=("smoke","screen","focused"),default="screen")
    p.add_argument("--config")
    p.add_argument("--out",required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--stop-after-epoch", type=int, help="Pause at this epoch while preserving the full scheduler horizon.")
    p.add_argument("--initialize")
    p.add_argument("--device")
    p = stage("infer",dataset=True,partition=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--device")
    p = stage("centerlines",dataset=True,partition=True)
    p.add_argument("--source",choices=("oracle","predicted"),required=True)
    p.add_argument("--predictions")
    p.add_argument("--out",required=True)
    p.add_argument("--aorta-dir")
    p.add_argument("--aorta-label",type=int,default=6)
    p.add_argument("--seeds")
    p.add_argument("--proxy",action="store_true")
    p.add_argument("--step-mm",type=float,default=.4)
    p.add_argument("--min-caliber-ratio",type=float,default=1.20)
    p = stage("smpr",dataset=True)
    p.add_argument("--centerlines",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--cross-mm",type=float,default=.4)
    p.add_argument("--fov-mm",type=float,default=20.)
    p.add_argument("--cross-pixels",type=int)
    p.add_argument("--long-axis",type=int,choices=(0,1,2),default=2)
    p.add_argument("--include-labels",action="store_true")
    p.add_argument("--overlays",action="store_true")
    p.add_argument("--frozen")
    p = stage("smpr-dataset",dataset=True)
    p.add_argument("--indices",nargs="+",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--frozen")
    p = stage("moosavi-fetch")
    p.add_argument("--out",required=True)
    p = stage("moosavi-inspect")
    p.add_argument("--model",required=True)
    p.add_argument("--samples",required=True)
    p.add_argument("--out",required=True)
    p = stage("moosavi-inputs")
    p.add_argument("--index",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--intensity-offset",type=float,required=True)
    p = stage("moosavi-infer")
    p.add_argument("--model",required=True)
    p.add_argument("--images",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--folds",nargs="+",type=int,default=[0])
    p.add_argument("--checkpoint",default="checkpoint_final.pth")
    p.add_argument("--device",default="cuda")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--mirror",action="store_true")
    p.add_argument("--frozen")
    p.add_argument("--dataset")
    p = stage("moosavi-verify")
    for name in ("model","samples","scores","out"):
        p.add_argument("--"+name,required=True)
    p = stage("inverse",dataset=True)
    for name in ("index","scores","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--plaque-channel",type=int,default=2)
    p.add_argument("--threshold",type=float,default=.5)
    p.add_argument("--engine",choices=("nnunet","xyz"),default="nnunet")
    p.add_argument("--initialization",choices=("pretrained","fine-tuned","random"),required=True)
    p.add_argument("--frozen")
    p = stage("nnunet-export",dataset=True)
    p.add_argument("--raw",required=True)
    p.add_argument("--name",default="Dataset701_SHKPlaque")
    p.add_argument("--task",choices=tuple(TASKS),default="multiclass")
    p.add_argument("--intensity-offset",type=float,default=0.)
    p = stage("nnunet-transfer-plans")
    for name in ("model","exported","preprocessed"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--name",default="MoosaviTransfer")
    p = stage("nnunet-split")
    p.add_argument("--exported",required=True)
    p.add_argument("--preprocessed",required=True)
    p = stage("nnunet-install-trainers")
    p.add_argument("--directory")
    p = stage("nnunet-inputs",dataset=True,partition=True)
    p.add_argument("--out",required=True)
    p.add_argument("--intensity-offset",type=float,default=0.)
    p = stage("nnunet-import",dataset=True,partition=True)
    for name in ("predictions","out","checkpoint"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--task",choices=tuple(TASKS),default="multiclass")
    p.add_argument("--probability-axes",choices=("xyz","zyx"),default="zyx")
    p.add_argument("--initialization",choices=("pretrained","fine-tuned","random"),required=True)
    p = stage("evaluate",dataset=True,partition=True)
    p.add_argument("--predictions",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--surface-tolerance-mm",type=float,default=1.)
    p.add_argument("--lesion-iou",type=float,default=0.)
    p.add_argument("--threshold",type=float)
    p.add_argument("--assessment-from")
    p = stage("ensemble",dataset=True,partition=True)
    p.add_argument("--predictions",nargs="+",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--method",choices=("soft","vote","confidence","rule","learned"),default="soft")
    p.add_argument("--weights",type=float,nargs="+")
    p.add_argument("--gating")
    p.add_argument("--threshold",type=float,default=.5)
    p = stage("specialists",dataset=True,partition=True)
    for name in ("lumen-predictions","plaque-predictions","out"):
        p.add_argument("--"+name,required=True)
    p = stage("fit-gate",dataset=True)
    p.add_argument("--predictions",nargs="+",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=2028)
    p.add_argument("--samples-per-case",type=int,default=12000)
    p = stage("postprocess",dataset=True,partition=True)
    p.add_argument("--predictions",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--min-volume-mm3",type=float,default=0.)
    p.add_argument("--threshold",type=float,default=.5)
    p.add_argument("--lumen-predictions")
    p.add_argument("--max-lumen-distance-mm",type=float)
    p.add_argument("--hu-min",type=float)
    p.add_argument("--hu-max",type=float)
    p = stage("freeze",dataset=True)
    p.add_argument("--checkpoints",nargs="*",default=[])
    p.add_argument("--runs",nargs="*",default=[])
    p.add_argument("--decision",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--surface-tolerance-mm",type=float,default=1.)
    p.add_argument("--lesion-iou",type=float,default=0.)
    p = stage("geometry-metrics",dataset=True)
    for name in ("centerlines","predictions","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--frozen")
    p = stage("compare")
    p.add_argument("--reports",nargs="+",required=True)
    p.add_argument("--out",required=True)
    p = stage("run")
    p.add_argument("--spec",required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    stage = args.stage
    manifest = read_json(args.dataset) if getattr(args,"dataset",None) else None
    if getattr(args,"partition",None):
        guard(manifest,args.partition,args.frozen)
    if stage == "synthetic":
        from .synthetic import create
        create(args.out,args.cases)
    elif stage == "audit":
        from .evaluation import _csv
        result = audit(args.images,args.masks,args.review)
        write_json(args.out,result,lock=True)
        _csv(Path(args.out).with_suffix(".csv"),[{"case":r["case"],"status":r["status"],"review":r["review"],
            "errors":" | ".join(r["errors"]),"flags":" | ".join(r["flags"]),
            **{f"{b}_plaque_voxels":r["branches"].get(b,{}).get("plaque_voxels") for b in ("LAD","LCX","RCA")}}
            for r in result["cases"]])
    elif stage == "split":
        write_json(args.out,make_split(read_json(args.audit),args.seed,require_reviewed=args.require_reviewed),lock=True)
    elif stage == "prepare":
        prepare(read_json(args.audit),read_json(args.split),args.out)
    elif stage == "train":
        from .native import config_for, train
        train(manifest,args.task,config_for(args.model,args.profile,args.config),args.out,
              device=args.device,resume=args.resume,initialize=args.initialize,stop_after_epoch=args.stop_after_epoch)
    elif stage == "infer":
        from .native import infer
        guard(manifest,args.partition,args.frozen,file_sha256(args.checkpoint))
        infer(manifest,args.partition,args.checkpoint,args.out,device=args.device)
    elif stage == "centerlines":
        from .geometry import batch_centerlines
        batch_centerlines(manifest,args.partition,args.out,args.source,predictions=args.predictions,
            aorta_dir=args.aorta_dir,aorta_label=args.aorta_label,seeds=args.seeds,proxy=args.proxy,
            step_mm=args.step_mm,min_ratio=args.min_caliber_ratio)
    elif stage == "smpr":
        from .smpr import batch_reconstruct
        index = read_json(args.centerlines)
        guard(manifest,index["partition"],args.frozen)
        batch_reconstruct(manifest,index,args.out,args.cross_mm,args.fov_mm,args.long_axis,args.overlays,
                          args.include_labels,args.cross_pixels)
    elif stage == "smpr-dataset":
        from .external import smpr_manifest
        indices = [read_json(path) for path in args.indices]
        for index in indices:
            guard(manifest,index["partition"],args.frozen)
        smpr_manifest(manifest,indices,args.out)
    elif stage == "moosavi-fetch":
        from .moosavi import fetch_public_assets
        fetch_public_assets(args.out)
    elif stage == "moosavi-inspect":
        from .moosavi import inspect_assets
        inspect_assets(args.model,args.samples,args.out)
    elif stage == "moosavi-inputs":
        from .moosavi import prepare_inputs
        prepare_inputs(read_json(args.index),args.out,args.intensity_offset)
    elif stage == "moosavi-infer":
        from .moosavi import infer_folder
        stamp = Path(args.images)/"reconstruction_provenance.json"
        if stamp.is_file() and read_json(stamp)["partition"] == "test":
            if manifest is None:
                raise ValueError("Supply --dataset for protected test inference.")
            for fold in args.folds:
                guard(manifest,"test",args.frozen,file_sha256(Path(args.model)/f"fold_{fold}"/args.checkpoint))
        infer_folder(args.model,args.images,args.out,folds=args.folds,checkpoint=args.checkpoint,
                     device=args.device,cpu_threads=args.threads,mirror=args.mirror)
    elif stage == "moosavi-verify":
        from .moosavi import verify_public_predictions
        verify_public_predictions(args.model,args.samples,args.scores,args.out)
    elif stage == "inverse":
        from .smpr import inverse_batch
        index = read_json(args.index)
        guard(manifest,index["partition"],args.frozen)
        inverse_batch(manifest,index,args.scores,args.out,plaque_channel=args.plaque_channel,
                      threshold=args.threshold,engine=args.engine,initialization=args.initialization)
    elif stage == "nnunet-export":
        from .external import native_export
        native_export(manifest,args.raw,args.name,task=args.task,intensity_offset=args.intensity_offset)
    elif stage == "nnunet-transfer-plans":
        from .external import transfer_plans
        transfer_plans(args.model,args.exported,args.preprocessed,args.name)
    elif stage == "nnunet-split":
        from .external import install_split
        install_split(args.exported,args.preprocessed)
    elif stage == "nnunet-install-trainers":
        from .external import install_trainers
        install_trainers(args.directory)
    elif stage == "nnunet-inputs":
        from .external import export_inputs
        export_inputs(manifest,args.partition,args.out,args.intensity_offset)
    elif stage == "nnunet-import":
        from .external import import_nnunet
        guard(manifest,args.partition,args.frozen,file_sha256(args.checkpoint))
        import_nnunet(manifest,args.partition,args.predictions,args.out,args.task,
                      probability_axes=args.probability_axes,checkpoint=args.checkpoint,initialization=args.initialization)
    elif stage == "evaluate":
        from .prediction import read_run
        from .evaluation import evaluate
        meta = read_run(args.predictions,manifest,args.partition)
        guard(manifest,args.partition,args.frozen,meta.get("candidate_id") or "unidentified")
        if args.partition == "test":
            protocol = read_json(args.frozen)["evaluation_protocol"]
            if protocol != {"surface_tolerance_mm":args.surface_tolerance_mm,"lesion_iou":args.lesion_iou} or args.threshold is not None:
                raise ValueError("Promote matching evaluation settings and thresholded validation outputs before test use.")
        evaluate(manifest,args.partition,args.predictions,args.out,surface_tolerance=args.surface_tolerance_mm,
                 lesion_iou=args.lesion_iou,threshold=args.threshold,assessment_from=args.assessment_from)
    elif stage == "ensemble":
        from .fusion import ensemble
        ensemble(manifest,args.partition,args.predictions,args.out,method=args.method,weights=args.weights,
                 gating=args.gating,threshold=args.threshold,frozen=args.frozen)
    elif stage == "specialists":
        from .fusion import specialists
        specialists(manifest,args.partition,args.lumen_predictions,args.plaque_predictions,args.out,args.frozen)
    elif stage == "fit-gate":
        from .fusion import fit_gate
        fit_gate(manifest,args.predictions,args.out,seed=args.seed,per_case=args.samples_per_case)
    elif stage == "postprocess":
        from .fusion import refine
        refine(manifest,args.partition,args.predictions,args.out,min_volume_mm3=args.min_volume_mm3,
               threshold=args.threshold,lumen_root=args.lumen_predictions,max_lumen_distance_mm=args.max_lumen_distance_mm,
               hu_min=args.hu_min,hu_max=args.hu_max,frozen=args.frozen)
    elif stage == "freeze":
        from .protection import freeze
        freeze(manifest,args.out,checkpoints=args.checkpoints,runs=args.runs,decision=args.decision,
               surface_tolerance_mm=args.surface_tolerance_mm,lesion_iou=args.lesion_iou)
    elif stage == "geometry-metrics":
        from .geometry_metrics import geometry_metrics
        index = read_json(args.centerlines)
        guard(manifest,index["partition"],args.frozen)
        geometry_metrics(manifest,index,args.predictions,args.out)
    elif stage == "compare":
        from .reporting import compare
        compare(args.reports,args.out)
    elif stage == "run":
        import subprocess
        import sys
        steps = read_json(args.spec)
        if not isinstance(steps,list) or not all(isinstance(step,list) and all(isinstance(v,str) for v in step) for step in steps):
            raise ValueError("Runner specification must contain string argument lists.")
        for step in steps:
            if step and step[0] == "run":
                raise ValueError("Nested runners are unsupported.")
            subprocess.run([sys.executable,"-m","plaque_benchmark",*step],check=True)
    print(f"Completed {stage}",flush=True)


if __name__ == "__main__":
    main()

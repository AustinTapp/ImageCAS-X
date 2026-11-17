"""Generate comparison tables and figures from measured outputs."""
from pathlib import Path
import csv

import numpy as np

from .data import read_json, write_json


def compare(reports, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reports,out = [Path(path) for path in reports],Path(out)
    summaries = [read_json(path/"summary.json") for path in reports]
    signatures = {(row["run"]["split_sha256"],row["run"]["partition"],tuple(row["evaluated_ids"])) for row in summaries}
    if len(signatures) != 1:
        raise ValueError("Comparison requires the same split, partition and assessment patients.")
    out.mkdir(parents=True,exist_ok=True)
    table = []
    for path,summary in zip(reports,summaries):
        run = summary["run"]
        plaque,healthy = summary.get("plaque",{}),summary.get("healthy_branches",{})
        gpu = run.get("gpu_hours")
        if run.get("checkpoint"):
            history = Path(run["checkpoint"]).parent/"history.csv"
            if history.is_file():
                with history.open(newline="") as stream:
                    values = list(csv.DictReader(stream))
                gpu = float(values[-1]["gpu_hours_cumulative"]) if values[-1]["gpu_hours_cumulative"] else None
        table.append({"name":run.get("display_name") or path.name,"model":run["model"],"geometry":run["geometry"],
            "initialization":run.get("initialization"),"plaque_dice":plaque.get("dice"),"plaque_precision":plaque.get("precision"),
            "plaque_recall":plaque.get("recall"),"healthy_branch_fps":healthy.get("fp_lesions_per_branch"),
            "healthy_branch_fp_rate":healthy.get("fp_rate"),"lumen_dice":summary.get("lumen",{}).get("dice"),
            "gpu_hours":gpu,"assessment_cases":summary["cases"]})
    with (out/"comparison.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(out/"comparison.json",table)
    names = [row["name"] for row in table]
    fig,axes = plt.subplots(1,3,figsize=(13,max(3.5,.42*len(table)+1.5)))
    for ax,key,title in zip(axes,("plaque_dice","healthy_branch_fp_rate","lumen_dice"),
                             ("Positive-case plaque Dice","Healthy-branch FP rate","Positive-case lumen Dice")):
        values = [np.nan if row[key] is None else row[key] for row in table]
        ax.barh(names,values)
        for y,value in enumerate(values):
            text = "undefined" if np.isnan(value) else f"{value:.3f}"
            ax.text(.02 if np.isnan(value) else min(value+.02,.86),y,text,va="center",fontsize=8)
        ax.set(xlim=(0,1),xlabel=title)
    fig.tight_layout()
    fig.savefig(out/"comparison.png",dpi=160)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(6,5))
    for row in table:
        if row["plaque_recall"] is not None and row["plaque_precision"] is not None:
            ax.scatter(row["plaque_recall"],row["plaque_precision"],label=row["name"])
    ax.set(xlabel="Plaque recall at declared threshold",ylabel="Plaque precision at declared threshold",xlim=(0,1),ylim=(0,1))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    else:
        ax.text(.5,.5,"No defined precision–recall pairs",ha="center",transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out/"precision_recall_operating_points.png",dpi=160)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,5))
    for row in table:
        if row["gpu_hours"] is not None and row["plaque_dice"] is not None:
            ax.scatter(row["gpu_hours"],row["plaque_dice"],label=row["name"])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    else:
        ax.text(.5,.5,"GPU timing unavailable for these runs",ha="center",transform=ax.transAxes)
    ax.set(xlabel="Measured GPU hours",ylabel="Positive-case plaque Dice",ylim=(0,1))
    fig.tight_layout()
    fig.savefig(out/"performance_gpu_hours.png",dpi=160)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,5))
    bins = [0,5,10,25,50,np.inf]
    labels = ["<5","5–10","10–25","25–50","≥50"]
    for path,row in zip(reports,table):
        lesions = read_json(path/"lesions.json")
        sensitivity = []
        for lower,upper in zip(bins[:-1],bins[1:]):
            selected = [item for item in lesions if lower <= item["volume_mm3"] < upper]
            sensitivity.append(np.mean([item["detected"] for item in selected]) if selected else np.nan)
        ax.plot(labels,sensitivity,marker="o",label=row["name"])
    ax.set(xlabel="Ground-truth component volume (mm³)",ylabel="Lesion detection sensitivity",ylim=(0,1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"lesion_size_sensitivity.png",dpi=160)
    plt.close(fig)
    ids = summaries[0]["evaluated_ids"]
    scores = []
    for path in reports:
        values = {r["case"]:r["dice"] for r in read_json(path/"metrics.json")
                  if r["scope"] == "end_to_end" and r["class"] == "plaque" and r["branch"] == "ALL"}
        scores.append([np.nan if values[case] is None else values[case] for case in ids])
    fig,ax = plt.subplots(figsize=(max(7,.22*len(ids)),max(3,.4*len(names)+1)))
    image = ax.imshow(scores,vmin=0,vmax=1,aspect="auto")
    ax.set_yticks(range(len(names)),names)
    ax.set_xticks(range(len(ids)),ids,rotation=90,fontsize=6)
    ax.set_title("Positive-case plaque Dice and common failure cases")
    fig.colorbar(image,ax=ax)
    fig.tight_layout()
    fig.savefig(out/"case_failure_matrix.png",dpi=160)
    plt.close(fig)
    return table

from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# CONFIGURATION
# ============================================================

SEGMENTATION_DIR = Path(
    r"D:\Data\ImageCAS\MultiX\segmentations"
)

OUTPUT_CSV = SEGMENTATION_DIR.parent / "MultiX_label_case_prevalence.csv"

EXPECTED_NUM_CASES = 800

LABELS = {
    0: "Background",
    1: "Myocardium",
    2: "LA",
    3: "LV",
    4: "RA",
    5: "RV",
    6: "Aorta",
    7: "PA",
    8: "LAA",
    9: "PV",
    10: "LM",
    11: "LAD",
    12: "LCx",
    13: "D1",
    14: "D2",
    15: "OM1",
    16: "OM2",
    17: "IM",
    18: "RCA",
    19: "R-PDA",
    20: "R-PLA",
    21: "L-PDA",
    22: "L-PLA",
    23: "Other",
}

# Normally only foreground classes are useful for prevalence analysis.
INCLUDE_BACKGROUND = False


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        SEGMENTATION_DIR.glob("*.nii.gz"),
        key=lambda p: int(p.name.split(".")[0])
        if p.name.split(".")[0].isdigit()
        else p.name
    )

    if len(files) == 0:
        raise RuntimeError(
            f"No .nii.gz files found in:\n{SEGMENTATION_DIR}"
        )

    print(f"Found {len(files)} segmentation files.")

    if EXPECTED_NUM_CASES is not None and len(files) != EXPECTED_NUM_CASES:
        print(
            f"[WARNING] Expected {EXPECTED_NUM_CASES} cases, "
            f"but found {len(files)}."
        )

    label_ids = sorted(LABELS.keys())

    if not INCLUDE_BACKGROUND:
        label_ids = [x for x in label_ids if x != 0]

    # Per-label storage
    stats = {}

    for label_id in label_ids:
        stats[label_id] = {
            "positive_cases": 0,
            "voxel_counts": [],
            "physical_volumes_mm3": [],
        }

    # --------------------------------------------------------
    # Process cases
    # --------------------------------------------------------

    for i, path in enumerate(files, start=1):

        image = sitk.ReadImage(str(path))
        arr = sitk.GetArrayViewFromImage(image)

        # mm in SimpleITK: x, y, z
        spacing = image.GetSpacing()
        voxel_volume_mm3 = float(np.prod(spacing))

        unique_labels, counts = np.unique(arr, return_counts=True)
        case_counts = dict(zip(unique_labels.astype(int), counts.astype(int)))

        # Check for unexpected labels
        unexpected = sorted(
            set(case_counts.keys()) - set(LABELS.keys())
        )

        if unexpected:
            print(
                f"[WARNING] {path.name}: unexpected labels {unexpected}"
            )

        for label_id in label_ids:

            n_voxels = case_counts.get(label_id, 0)

            if n_voxels > 0:

                stats[label_id]["positive_cases"] += 1
                stats[label_id]["voxel_counts"].append(n_voxels)

                volume_mm3 = n_voxels * voxel_volume_mm3
                stats[label_id]["physical_volumes_mm3"].append(
                    volume_mm3
                )

        if i % 50 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)} cases")

    # --------------------------------------------------------
    # Construct output table
    # --------------------------------------------------------

    total_cases = len(files)

    rows = []

    for label_id in label_ids:

        s = stats[label_id]

        positive_cases = s["positive_cases"]
        negative_cases = total_cases - positive_cases

        voxel_counts = np.asarray(
            s["voxel_counts"],
            dtype=np.float64
        )

        volumes = np.asarray(
            s["physical_volumes_mm3"],
            dtype=np.float64
        )

        if positive_cases > 0:

            total_voxels = int(voxel_counts.sum())

            mean_voxels = float(voxel_counts.mean())
            median_voxels = float(np.median(voxel_counts))
            min_voxels = int(voxel_counts.min())
            max_voxels = int(voxel_counts.max())

            total_volume_mm3 = float(volumes.sum())
            mean_volume_mm3 = float(volumes.mean())
            median_volume_mm3 = float(np.median(volumes))

        else:

            total_voxels = 0

            mean_voxels = 0.0
            median_voxels = 0.0
            min_voxels = 0
            max_voxels = 0

            total_volume_mm3 = 0.0
            mean_volume_mm3 = 0.0
            median_volume_mm3 = 0.0

        prevalence = positive_cases / total_cases

        rows.append(
            {
                "label_id": label_id,
                "label_name": LABELS[label_id],

                "total_cases": total_cases,
                "positive_cases": positive_cases,
                "negative_cases": negative_cases,

                "case_prevalence_fraction": prevalence,
                "case_prevalence_percent": prevalence * 100.0,

                "total_voxels": total_voxels,

                "mean_voxels_positive_case": mean_voxels,
                "median_voxels_positive_case": median_voxels,
                "min_voxels_positive_case": min_voxels,
                "max_voxels_positive_case": max_voxels,

                "total_volume_mm3": total_volume_mm3,
                "mean_volume_mm3_positive_case": mean_volume_mm3,
                "median_volume_mm3_positive_case": median_volume_mm3,

                "mean_volume_ml_positive_case":
                    mean_volume_mm3 / 1000.0,

                "median_volume_ml_positive_case":
                    median_volume_mm3 / 1000.0,
            }
        )

    df = pd.DataFrame(rows)

    # Preserve anatomical/label order in main output.
    df.to_csv(OUTPUT_CSV, index=False)

    # --------------------------------------------------------
    # Terminal summary sorted rare -> common
    # --------------------------------------------------------

    display_df = df[
        [
            "label_id",
            "label_name",
            "positive_cases",
            "case_prevalence_percent",
            "median_voxels_positive_case",
            "median_volume_mm3_positive_case",
        ]
    ].sort_values(
        ["positive_cases", "label_id"],
        ascending=[True, True]
    )

    print("\n")
    print("=" * 90)
    print("LABEL PREVALENCE — RAREST TO MOST COMMON")
    print("=" * 90)

    print(
        display_df.to_string(
            index=False,
            formatters={
                "case_prevalence_percent":
                    lambda x: f"{x:6.2f}%",

                "median_voxels_positive_case":
                    lambda x: f"{x:,.0f}",

                "median_volume_mm3_positive_case":
                    lambda x: f"{x:,.2f}",
            },
        )
    )

    print("\nSaved:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
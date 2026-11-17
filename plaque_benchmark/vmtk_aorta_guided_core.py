#!/usr/bin/env python3
r"""
Batch coronary centerline extraction for the ImageCAS processing layout.

Default inputs
--------------
Raw multi-label nnU-Net segmentations (aorta label = 6):
    D:\Data\ImageCAS\Multilabel\segmentations\1.img.nii.gz
    D:\Data\ImageCAS\Multilabel\segmentations\2.img.nii.gz
    ...

Processed binary coronary segmentations:
    D:\Data\ImageCAS\Multilabel\CAE\1.img\left_coronary.nii.gz
    D:\Data\ImageCAS\Multilabel\CAE\1.img\right_coronary.nii.gz
    D:\Data\ImageCAS\Multilabel\CAE\2.img\...

For each left/right coronary segmentation the script:
1. Generates the working coronary surface in memory (no STL is written).
2. Finds a centered aorta-coronary inlet anchor from the ostial interface patch.
3. Uses cumulative dilation only to stabilize the number of distal endpoints.
4. Snaps the final inlet/outlets back to the original non-dilated surface.
5. Computes source-to-target VMTK centerlines on that original surface.

Saved outputs (beside each processed coronary segmentation)
-----------------------------------------------------------
    left_coronary_centerline.vtp
    left_coronary_centerline_01.vtp
    left_coronary_centerline_02.vtp
    ...
    left_coronary_endpoints.mrk.json

and equivalently for right_coronary.

The numbered VTP files contain the individual source-to-target centerline cells
that form the combined centerline. The Slicer markup JSON contains the final
inlet as the first control point followed by all final outlet points.

Not saved
---------
- source image copies
- segmentation copies
- generated Flying Edges STL meshes
- Voronoi diagram
- endpoint-history CSV
- stable-endpoint network
- intermediate dilation meshes/networks

Coordinates are written in the unmodified SimpleITK physical coordinate system,
normally LPS for medical images.

Required packages: numpy, SimpleITK, vtk, vmtk

Example
-------
python extract_centerline_aorta_guided_ImageCAS.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import SimpleITK as sitk
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


DEFAULT_INPUT_ROOT = Path(r"D:\Data\ImageCAS\Multilabel\CAE")
DEFAULT_AORTA_ROOT = Path(r"D:\Data\ImageCAS\Multilabel\segmentations")
DEFAULT_INPUT_FILENAMES = (
    "left_coronary.nii.gz",
    "right_coronary.nii.gz",
)
DEFAULT_FOREGROUND_LABEL = 1
DEFAULT_AORTA_LABEL = 6
DEFAULT_MAX_DILATIONS = 5
DEFAULT_STABLE_COMPARISONS = 1
DEFAULT_DILATION_RADIUS = (1, 1, 1)
DEFAULT_DILATION_KERNEL = "box"

# These defaults match the attached Flying Edges surface mesher.
DEFAULT_RETAINED_FRACTION = 0.25
DEFAULT_SMOOTHING_ITERATIONS = 15
DEFAULT_PASS_BAND = 0.001
DEFAULT_FEATURE_ANGLE = 120.0

# These defaults mirror the Slicer ExtractCenterline preprocessing intent.
DEFAULT_ENDPOINT_TARGET_POINTS = 5000
DEFAULT_CURVE_SAMPLING_DISTANCE = 1.0

# Centered ostium/interface localization. The initial band is measured outward
# from the minimum coronary-boundary-to-aorta distance. It is expanded only
# when too few boundary voxels are available to define a stable interface patch.
DEFAULT_INLET_INTERFACE_BAND_MM = 0.75
DEFAULT_INLET_INTERFACE_MIN_VOXELS = 12
DEFAULT_INLET_INTERFACE_MAX_BAND_MM = 2.0

RADIUS_ARRAY_NAME = "Radius"
TOPOLOGY_ARRAY_NAME = "Topology"
MARKS_ARRAY_NAME = "Marks"


@dataclass(frozen=True)
class Configuration:
    input_root: Path
    aorta_root: Path
    input_filenames: tuple[str, ...]
    direct_inputs: tuple[Path, ...]
    aorta_image: Path | None
    foreground_label: int
    aorta_label: int
    max_dilations: int
    stable_comparisons: int
    dilation_radius: tuple[int, int, int]
    dilation_kernel: str
    retained_fraction: float
    smoothing_iterations: int
    pass_band: float
    feature_angle: float
    endpoint_target_points: int
    curve_sampling_distance: float
    inlet_interface_band_mm: float
    inlet_interface_min_voxels: int
    inlet_interface_max_band_mm: float
    simplify_voronoi: bool
    overwrite: bool
    verbose: bool


@dataclass
class EndpointDetection:
    dilation_iteration: int
    endpoint_positions: list[tuple[float, float, float]]
    inlet_coronary_position: tuple[float, float, float]
    inlet_aorta_position: tuple[float, float, float]
    inlet_distance_mm: float
    network: vtk.vtkPolyData
    surface: vtk.vtkPolyData
    foreground_voxels: int

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_positions)


@dataclass
class HistoryRecord:
    dilation_iteration: int
    endpoint_count: int
    unchanged_from_previous: bool
    stable_comparison_run: int
    foreground_voxels: int
    mesh_points: int
    mesh_triangles: int
    source_x: float | None
    source_y: float | None
    source_z: float | None
    inlet_anchor_x: float
    inlet_anchor_y: float
    inlet_anchor_z: float
    aorta_x: float
    aorta_y: float
    aorta_z: float
    aorta_distance_mm: float


@dataclass(frozen=True)
class InletProximity:
    coronary_position: tuple[float, float, float]
    aorta_position: tuple[float, float, float]
    distance_mm: float
    coronary_point_id: int


@dataclass
class OutputPaths:
    centerline: Path
    endpoints_markup: Path


class EndpointStabilityError(RuntimeError):
    """Raised when endpoint-count stability is not reached."""

    def __init__(
        self,
        message: str,
        history: Sequence[HistoryRecord],
    ) -> None:
        super().__init__(message)
        self.history = list(history)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compute VMTK centerlines using endpoints whose count stabilizes "
            "across repeated 1x1x1 segmentation dilations."
        ),
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing per-case folders (for example 1.img, 2.img) with left/right coronary masks.",
    )
    parser.add_argument(
        "--aorta-root",
        type=Path,
        default=DEFAULT_AORTA_ROOT,
        help=(
            "Root containing flat raw multi-label segmentations, such as "
            "1.img.nii.gz and 2.img.nii.gz, with aorta label 6."
        ),
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=list(DEFAULT_INPUT_FILENAMES),
        help="Segmentation filenames expected inside each per-case folder.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "One or more explicit segmentation paths. When supplied, CaseX "
            "directory scanning is disabled."
        ),
    )
    parser.add_argument(
        "--aorta-image",
        type=Path,
        default=None,
        help=(
            "Explicit multi-label image containing the aorta. Valid only "
            "when exactly one direct --inputs segmentation is supplied."
        ),
    )
    parser.add_argument(
        "--label",
        type=int,
        default=DEFAULT_FOREGROUND_LABEL,
        help="Foreground label in the finalized segmentation.",
    )
    parser.add_argument(
        "--aorta-label",
        type=int,
        default=DEFAULT_AORTA_LABEL,
        help="Aorta label in the corresponding multi-label segmentation.",
    )
    parser.add_argument(
        "--max-dilations",
        type=int,
        default=DEFAULT_MAX_DILATIONS,
        help=(
            "Maximum number of cumulative one-step dilations. Centerline "
            "extraction stops earlier once endpoint-count stability is found."
        ),
    )
    parser.add_argument(
        "--stable-comparisons",
        type=int,
        default=DEFAULT_STABLE_COMPARISONS,
        help=(
            "Required consecutive unchanged endpoint-count comparisons. A "
            "value of 1 accepts the first dilation whose count equals the "
            "immediately preceding count."
        ),
    )
    parser.add_argument(
        "--dilation-radius",
        nargs=3,
        type=int,
        metavar=("RX", "RY", "RZ"),
        default=list(DEFAULT_DILATION_RADIUS),
        help=(
            "Voxel radius applied at each cumulative dilation step. The "
            "default expands by one voxel along x, y, and z; with 0.3 x "
            "0.3 x 0.5 mm spacing this is one 0.3 x 0.3 x 0.5 mm step."
        ),
    )
    parser.add_argument(
        "--dilation-kernel",
        choices=("ball", "box", "cross"),
        default=DEFAULT_DILATION_KERNEL,
        help="SimpleITK binary-dilation structuring element.",
    )

    parser.add_argument(
        "--retained-fraction",
        type=float,
        default=DEFAULT_RETAINED_FRACTION,
        help="Requested triangle fraction retained for each dilated mesh.",
    )
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=DEFAULT_SMOOTHING_ITERATIONS,
        help="Windowed-sinc smoothing iterations for dilated meshes.",
    )
    parser.add_argument(
        "--pass-band",
        type=float,
        default=DEFAULT_PASS_BAND,
        help="Windowed-sinc pass band; smaller values smooth more strongly.",
    )
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=DEFAULT_FEATURE_ANGLE,
        help="Feature angle in degrees for windowed-sinc smoothing.",
    )
    parser.add_argument(
        "--endpoint-target-points",
        type=int,
        default=DEFAULT_ENDPOINT_TARGET_POINTS,
        help=(
            "Approximate maximum surface point count used during automatic "
            "network/endpoint extraction. Set 0 to disable this extra "
            "preprocessing decimation."
        ),
    )
    parser.add_argument(
        "--curve-sampling-distance",
        type=float,
        default=DEFAULT_CURVE_SAMPLING_DISTANCE,
        help="VMTK centerline resampling-step value in physical units.",
    )
    parser.add_argument(
        "--inlet-interface-band-mm",
        type=float,
        default=DEFAULT_INLET_INTERFACE_BAND_MM,
        help=(
            "Physical distance band, added to the minimum coronary-boundary-to-aorta "
            "distance, used to sample the ostial interface patch."
        ),
    )
    parser.add_argument(
        "--inlet-interface-min-voxels",
        type=int,
        default=DEFAULT_INLET_INTERFACE_MIN_VOXELS,
        help=(
            "Preferred minimum number of coronary boundary voxels in the selected "
            "ostial interface patch before its centroid is accepted."
        ),
    )
    parser.add_argument(
        "--inlet-interface-max-band-mm",
        type=float,
        default=DEFAULT_INLET_INTERFACE_MAX_BAND_MM,
        help=(
            "Maximum interface distance band used when automatically expanding the "
            "ostial patch to obtain enough boundary samples."
        ),
    )
    parser.add_argument(
        "--simplify-voronoi",
        action="store_true",
        help=(
            "Enable VMTK Voronoi simplification. Disabled by default because "
            "the original Slicer module disables it with modern VTK builds."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing centerline outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser


def parse_configuration(argv: Sequence[str] | None = None) -> Configuration:
    args = build_argument_parser().parse_args(argv)

    direct_inputs = tuple(args.inputs or ())
    dilation_radius = tuple(int(value) for value in args.dilation_radius)

    if args.aorta_image is not None and len(direct_inputs) != 1:
        raise ValueError(
            "--aorta-image requires exactly one segmentation supplied with "
            "--inputs."
        )
    if args.label < 0:
        raise ValueError("--label must be non-negative.")
    if args.aorta_label < 0:
        raise ValueError("--aorta-label must be non-negative.")
    if args.max_dilations < 1:
        raise ValueError("--max-dilations must be at least 1.")
    if args.stable_comparisons < 1:
        raise ValueError("--stable-comparisons must be at least 1.")
    if any(radius < 0 for radius in dilation_radius):
        raise ValueError("--dilation-radius values must be non-negative.")
    if dilation_radius == (0, 0, 0):
        raise ValueError("At least one --dilation-radius value must be positive.")
    if not 0.0 < args.retained_fraction <= 1.0:
        raise ValueError("--retained-fraction must be in (0, 1].")
    if args.smoothing_iterations < 0:
        raise ValueError("--smoothing-iterations must be non-negative.")
    if not 0.0 < args.pass_band <= 2.0:
        raise ValueError("--pass-band must be in (0, 2].")
    if not 0.0 <= args.feature_angle <= 180.0:
        raise ValueError("--feature-angle must be in [0, 180].")
    if args.endpoint_target_points < 0:
        raise ValueError("--endpoint-target-points must be non-negative.")
    if args.curve_sampling_distance <= 0.0:
        raise ValueError("--curve-sampling-distance must be positive.")
    if args.inlet_interface_band_mm <= 0.0:
        raise ValueError("--inlet-interface-band-mm must be positive.")
    if args.inlet_interface_min_voxels < 1:
        raise ValueError("--inlet-interface-min-voxels must be at least 1.")
    if args.inlet_interface_max_band_mm < args.inlet_interface_band_mm:
        raise ValueError(
            "--inlet-interface-max-band-mm must be greater than or equal to "
            "--inlet-interface-band-mm."
        )

    return Configuration(
        input_root=args.input_root,
        aorta_root=args.aorta_root,
        input_filenames=tuple(args.filenames),
        direct_inputs=direct_inputs,
        aorta_image=args.aorta_image,
        foreground_label=args.label,
        aorta_label=args.aorta_label,
        max_dilations=args.max_dilations,
        stable_comparisons=args.stable_comparisons,
        dilation_radius=dilation_radius,
        dilation_kernel=args.dilation_kernel,
        retained_fraction=args.retained_fraction,
        smoothing_iterations=args.smoothing_iterations,
        pass_band=args.pass_band,
        feature_angle=args.feature_angle,
        endpoint_target_points=args.endpoint_target_points,
        curve_sampling_distance=args.curve_sampling_distance,
        inlet_interface_band_mm=args.inlet_interface_band_mm,
        inlet_interface_min_voxels=args.inlet_interface_min_voxels,
        inlet_interface_max_band_mm=args.inlet_interface_max_band_mm,
        simplify_voronoi=args.simplify_voronoi,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


def load_vmtk_modules() -> SimpleNamespace:
    """Load modern standalone VMTK, with an older-wrapper fallback."""
    modern_error: Exception | None = None
    try:
        from vmtk import vtkvmtk  # type: ignore

        return SimpleNamespace(computational=vtkvmtk, misc=vtkvmtk)
    except Exception as exc:  # ImportError and platform DLL-load failures
        modern_error = exc

    try:
        import vtkvmtkComputationalGeometryPython as computational  # type: ignore
        import vtkvmtkMiscPython as misc  # type: ignore

        return SimpleNamespace(computational=computational, misc=misc)
    except Exception as legacy_error:
        raise ImportError(
            "VMTK could not be imported. Install/configure a standalone VMTK "
            "environment that supports `from vmtk import vtkvmtk`. The older "
            "Slicer-style wrapped modules were also unavailable. Modern import "
            f"error: {modern_error!r}; legacy import error: {legacy_error!r}"
        ) from legacy_error


def natural_case_key(path: Path) -> tuple[object, ...]:
    """Sort Case2 before Case10."""
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def strip_medical_image_suffix(filename: str) -> str:
    lower_name = filename.lower()
    for suffix in (".nii.gz", ".nrrd", ".mha", ".mhd", ".nii"):
        if lower_name.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def resolve_segmentation_paths(config: Configuration) -> list[Path]:
    if config.direct_inputs:
        return list(config.direct_inputs)

    if not config.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {config.input_root}")

    case_directories = sorted(
        (
            path
            for path in config.input_root.iterdir()
            if path.is_dir()
            and any((path / filename).is_file() for filename in config.input_filenames)
        ),
        key=natural_case_key,
    )
    if not case_directories:
        raise FileNotFoundError(
            f"No case directories containing {config.input_filenames} were found under {config.input_root}"
        )

    paths: list[Path] = []
    for case_directory in case_directories:
        for filename in config.input_filenames:
            candidate = case_directory / filename
            if candidate.is_file():
                paths.append(candidate)
            else:
                logging.warning("Missing segmentation: %s", candidate)
    return paths


def is_medical_image_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith((".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"))


def images_have_matching_geometry(
    first: sitk.Image,
    second: sitk.Image,
    tolerance: float = 1e-5,
) -> bool:
    if first.GetDimension() != second.GetDimension():
        return False
    if first.GetSize() != second.GetSize():
        return False
    for first_values, second_values in (
        (first.GetSpacing(), second.GetSpacing()),
        (first.GetOrigin(), second.GetOrigin()),
        (first.GetDirection(), second.GetDirection()),
    ):
        if not np.allclose(
            np.asarray(first_values, dtype=float),
            np.asarray(second_values, dtype=float),
            rtol=0.0,
            atol=tolerance,
        ):
            return False
    return True


def read_label_as_binary(
    image_path: Path,
    label: int,
) -> tuple[sitk.Image, sitk.Image, int]:
    image = sitk.ReadImage(str(image_path))
    if image.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D label image, but {image_path.name} has "
            f"{image.GetDimension()} dimensions."
        )

    array = sitk.GetArrayFromImage(image)
    binary_array = np.asarray(array == label, dtype=np.uint8)
    voxel_count = int(np.count_nonzero(binary_array))
    if voxel_count == 0:
        raise ValueError(f"Label {label} is absent from {image_path}.")

    binary = sitk.GetImageFromArray(binary_array)
    binary.CopyInformation(image)
    return image, binary, voxel_count


def case_name_for_coronary_input(segmentation_path: Path) -> str:
    """Return the case identifier (for example ``1.img``) for a coronary mask."""
    parent_name = segmentation_path.parent.name
    if parent_name and parent_name != ".":
        return parent_name

    base_name = strip_medical_image_suffix(segmentation_path.name)
    if base_name:
        return base_name
    raise ValueError(f"Could not infer a case identifier from: {segmentation_path}")


def flat_case_aorta_candidates(aorta_root: Path, case_name: str) -> list[Path]:
    """Find flat raw multi-label files matching one case identifier."""
    if not aorta_root.is_dir():
        raise FileNotFoundError(f"Aorta-label root does not exist: {aorta_root}")

    case_name_lower = case_name.lower()
    preferred_names = (
        f"{case_name}.nii.gz",
        f"{case_name}.nii",
        f"{case_name}.nrrd",
        f"{case_name}.mha",
        f"{case_name}.mhd",
    )

    candidates: list[Path] = []
    for preferred_name in preferred_names:
        candidate = aorta_root / preferred_name
        if candidate.is_file():
            candidates.append(candidate)

    # Files are expected directly under aorta_root. Permit suffixes such as
    # Case1_segmentation.nii.gz, but never mix Case1 with Case10.
    case_token_pattern = re.compile(
        rf"^{re.escape(case_name_lower)}(?:$|[_ .-].*)",
        re.IGNORECASE,
    )
    discovered = sorted(
        (
            path
            for path in aorta_root.iterdir()
            if path.is_file()
            and is_medical_image_path(path)
            and case_token_pattern.match(
                strip_medical_image_suffix(path.name).lower()
            )
        ),
        key=lambda path: str(path).lower(),
    )
    candidates.extend(path for path in discovered if path not in candidates)
    return candidates


def resolve_aorta_label_image(
    segmentation_path: Path,
    coronary_reference: sitk.Image,
    config: Configuration,
) -> tuple[Path, sitk.Image, sitk.Image]:
    """
    Locate the flat raw multi-label image containing the requested aorta label.

    For a coronary input inside ``.../1.img/``, the preferred source is
    ``<aorta_root>/1.img.nii.gz``. Other supported extensions and filenames
    beginning with the exact CaseX token are accepted as fallbacks. Candidate
    files are ranked using geometry agreement, filename cues, and aorta voxel
    count.
    """
    if config.aorta_image is not None:
        candidate_paths = [config.aorta_image]
        case_name = strip_medical_image_suffix(config.aorta_image.name)
    else:
        case_name = case_name_for_coronary_input(segmentation_path)
        candidate_paths = flat_case_aorta_candidates(
            config.aorta_root,
            case_name,
        )

    if not candidate_paths:
        raise FileNotFoundError(
            "No candidate aorta label images were found for "
            f"{segmentation_path}."
        )

    ranked: list[
        tuple[int, int, str, Path, sitk.Image, sitk.Image]
    ] = []
    read_errors: list[str] = []

    for candidate in candidate_paths:
        try:
            aorta_reference, aorta_binary, voxel_count = read_label_as_binary(
                candidate,
                config.aorta_label,
            )
        except Exception as exc:
            read_errors.append(f"{candidate.name}: {exc}")
            continue

        score = 0
        lower_name = candidate.name.lower()
        candidate_base = strip_medical_image_suffix(candidate.name)
        if candidate_base.lower() == case_name.lower():
            score += 100
        if images_have_matching_geometry(coronary_reference, aorta_reference):
            score += 80
        if "69" in lower_name:
            score += 30
        if any(token in lower_name for token in ("seg", "label", "mask")):
            score += 20
        if "aorta" in lower_name:
            score += 10
        if any(token in lower_name for token in ("left_coronary", "right_coronary")):
            score -= 50

        ranked.append(
            (
                score,
                voxel_count,
                str(candidate).lower(),
                candidate,
                aorta_reference,
                aorta_binary,
            )
        )

    if not ranked:
        details = "; ".join(read_errors[:8])
        raise FileNotFoundError(
            f"No image containing aorta label {config.aorta_label} was found "
            f"for {case_name}. Checked "
            f"{len(candidate_paths)} candidate(s). {details}"
        )

    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, voxel_count, _, selected_path, selected_reference, selected_binary = ranked[0]

    if len(ranked) > 1 and ranked[1][0:2] == ranked[0][0:2]:
        logging.warning(
            "Multiple aorta-label images had the same top score. Selected %s; "
            "alternative was %s.",
            selected_path,
            ranked[1][3],
        )

    logging.info(
        "Selected aorta label image: %s (label=%d, voxels=%d, geometry match=%s)",
        selected_path,
        config.aorta_label,
        voxel_count,
        images_have_matching_geometry(coronary_reference, selected_reference),
    )
    return selected_path, selected_reference, selected_binary


def output_paths_for(segmentation_path: Path) -> OutputPaths:
    base_name = strip_medical_image_suffix(segmentation_path.name)
    return OutputPaths(
        centerline=segmentation_path.with_name(f"{base_name}_centerline.vtp"),
        endpoints_markup=segmentation_path.with_name(f"{base_name}_endpoints.mrk.json"),
    )


def individual_centerline_path(segmentation_path: Path, index: int) -> Path:
    base_name = strip_medical_image_suffix(segmentation_path.name)
    return segmentation_path.with_name(f"{base_name}_centerline_{index:02d}.vtp")


def ensure_outputs_available(
    segmentation_path: Path,
    paths: OutputPaths,
    overwrite: bool,
) -> None:
    base_name = strip_medical_image_suffix(segmentation_path.name)
    existing = [path for path in asdict(paths).values() if Path(path).exists()]
    existing.extend(segmentation_path.parent.glob(f"{base_name}_centerline_[0-9][0-9].vtp"))
    existing = sorted(set(Path(path) for path in existing))
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Output files already exist ({names}). Use --overwrite.")
    if overwrite:
        for path in segmentation_path.parent.glob(f"{base_name}_centerline_[0-9][0-9].vtp"):
            path.unlink(missing_ok=True)


def read_binary_segmentation(
    image_path: Path,
    foreground_label: int,
) -> tuple[sitk.Image, sitk.Image]:
    """Read a 3D segmentation and return its reference and binary image."""
    reference = sitk.ReadImage(str(image_path))
    if reference.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D segmentation, but {image_path.name} has "
            f"{reference.GetDimension()} dimensions."
        )

    array = sitk.GetArrayFromImage(reference)
    binary_array = np.asarray(array == foreground_label, dtype=np.uint8)
    if not np.any(binary_array):
        raise ValueError(
            f"Foreground label {foreground_label} is absent from {image_path}."
        )

    binary = sitk.GetImageFromArray(binary_array)
    binary.CopyInformation(reference)
    return reference, binary


def count_foreground_voxels(binary_image: sitk.Image) -> int:
    return int(np.count_nonzero(sitk.GetArrayViewFromImage(binary_image)))


def dilation_kernel_enum(kernel_name: str) -> int:
    mapping = {
        "ball": sitk.sitkBall,
        "box": sitk.sitkBox,
        "cross": sitk.sitkCross,
    }
    return mapping[kernel_name]


def dilate_binary_image(
    binary_image: sitk.Image,
    radius_xyz: tuple[int, int, int],
    kernel_name: str,
) -> sitk.Image:
    """Apply one cumulative binary dilation step."""
    dilater = sitk.BinaryDilateImageFilter()
    dilater.SetKernelRadius(list(radius_xyz))
    dilater.SetKernelType(dilation_kernel_enum(kernel_name))
    dilater.SetForegroundValue(1)
    dilated = dilater.Execute(binary_image)
    return sitk.Cast(dilated, sitk.sitkUInt8)


def binary_mask_to_vtk_index_image(
    binary_mask_zyx: np.ndarray,
) -> vtk.vtkImageData:
    if binary_mask_zyx.ndim != 3:
        raise ValueError("The binary mask must be three-dimensional.")

    size_z, size_y, size_x = binary_mask_zyx.shape
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(size_x, size_y, size_z)
    vtk_image.SetOrigin(0.0, 0.0, 0.0)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)

    flat_mask = np.ascontiguousarray(binary_mask_zyx).ravel(order="C")
    vtk_scalars = numpy_to_vtk(
        num_array=flat_mask,
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    vtk_scalars.SetName("BinaryMask")
    vtk_image.GetPointData().SetScalars(vtk_scalars)
    return vtk_image


def validate_nonempty_surface(surface: vtk.vtkPolyData, stage: str) -> None:
    if surface is None or surface.GetNumberOfPoints() == 0:
        raise RuntimeError(f"{stage} produced no mesh points.")
    if surface.GetNumberOfPolys() == 0:
        raise RuntimeError(f"{stage} produced no polygonal faces.")


def generate_discrete_flying_edges_surface(
    vtk_index_image: vtk.vtkImageData,
) -> vtk.vtkPolyData:
    contour = vtk.vtkDiscreteFlyingEdges3D()
    contour.SetInputData(vtk_index_image)
    contour.SetValue(0, 1)
    contour.ComputeNormalsOff()
    contour.ComputeGradientsOff()
    contour.ComputeScalarsOff()
    contour.Update()

    surface = vtk.vtkPolyData()
    surface.DeepCopy(contour.GetOutput())
    validate_nonempty_surface(surface, "Discrete Flying Edges")
    return surface


def sitk_index_to_physical_matrix(
    reference_image: sitk.Image,
) -> vtk.vtkMatrix4x4:
    origin = np.asarray(reference_image.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(reference_image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(
        reference_image.GetDirection(),
        dtype=np.float64,
    ).reshape(3, 3)
    linear = direction @ np.diag(spacing)

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, float(linear[row, column]))
        matrix.SetElement(row, 3, float(origin[row]))
    return matrix


def transform_surface_to_physical_space(
    surface_in_index_space: vtk.vtkPolyData,
    reference_image: sitk.Image,
) -> vtk.vtkPolyData:
    transform = vtk.vtkTransform()
    transform.SetMatrix(sitk_index_to_physical_matrix(reference_image))

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(surface_in_index_space)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    transformed = vtk.vtkPolyData()
    transformed.DeepCopy(transform_filter.GetOutput())
    validate_nonempty_surface(transformed, "Physical-space transform")
    return transformed


def triangulate_and_clean(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(surface)
    triangle_filter.PassLinesOff()
    triangle_filter.PassVertsOff()
    triangle_filter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(triangle_filter.GetOutputPort())
    cleaner.PointMergingOn()
    cleaner.Update()

    cleaned = vtk.vtkPolyData()
    cleaned.DeepCopy(cleaner.GetOutput())
    validate_nonempty_surface(cleaned, "Triangle/clean stage")
    return cleaned


def smooth_surface(
    surface: vtk.vtkPolyData,
    iterations: int,
    pass_band: float,
    feature_angle: float,
) -> vtk.vtkPolyData:
    if iterations == 0:
        unchanged = vtk.vtkPolyData()
        unchanged.DeepCopy(surface)
        return unchanged

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(pass_band)
    smoother.SetFeatureAngle(feature_angle)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.GenerateErrorScalarsOff()
    smoother.GenerateErrorVectorsOff()
    smoother.Update()

    smoothed = vtk.vtkPolyData()
    smoothed.DeepCopy(smoother.GetOutput())
    validate_nonempty_surface(smoothed, "Smoothing")
    return smoothed


def decimate_surface(
    surface: vtk.vtkPolyData,
    retained_fraction: float,
) -> vtk.vtkPolyData:
    if retained_fraction >= 1.0:
        unchanged = vtk.vtkPolyData()
        unchanged.DeepCopy(surface)
        return unchanged

    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(surface)
    decimator.SetTargetReduction(1.0 - retained_fraction)
    decimator.PreserveTopologyOn()
    decimator.SplittingOff()
    decimator.BoundaryVertexDeletionOff()
    decimator.Update()

    decimated = vtk.vtkPolyData()
    decimated.DeepCopy(decimator.GetOutput())
    validate_nonempty_surface(decimated, "Decimation")
    return decimated


def recompute_normals(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    validate_nonempty_surface(output, "Normal generation")
    return output


def binary_image_to_surface(
    binary_image: sitk.Image,
    config: Configuration,
) -> vtk.vtkPolyData:
    """Convert a binary image to a final physical-space Flying Edges mesh."""
    binary_array = np.asarray(
        sitk.GetArrayFromImage(binary_image),
        dtype=np.uint8,
    )
    vtk_index_image = binary_mask_to_vtk_index_image(binary_array)
    index_surface = generate_discrete_flying_edges_surface(vtk_index_image)
    physical_surface = transform_surface_to_physical_space(
        index_surface,
        binary_image,
    )
    cleaned = triangulate_and_clean(physical_surface)
    smoothed = smooth_surface(
        cleaned,
        config.smoothing_iterations,
        config.pass_band,
        config.feature_angle,
    )
    decimated = decimate_surface(smoothed, config.retained_fraction)
    return recompute_normals(decimated)


def binary_image_to_reference_surface(
    binary_image: sitk.Image,
) -> vtk.vtkPolyData:
    """
    Convert a binary label image to an unsmoothed physical-space surface.

    This is used for the aorta because inlet localization should reflect the
    original label boundary rather than a smoothed or decimated approximation.
    """
    binary_array = np.asarray(
        sitk.GetArrayFromImage(binary_image),
        dtype=np.uint8,
    )
    vtk_index_image = binary_mask_to_vtk_index_image(binary_array)
    index_surface = generate_discrete_flying_edges_surface(vtk_index_image)
    physical_surface = transform_surface_to_physical_space(
        index_surface,
        binary_image,
    )
    return recompute_normals(triangulate_and_clean(physical_surface))


def geometry_matches(a: sitk.Image, b: sitk.Image, tolerance: float = 1.0e-6) -> bool:
    """Return whether two SimpleITK images share the same physical geometry."""
    if a.GetDimension() != b.GetDimension() or a.GetSize() != b.GetSize():
        return False
    for av, bv in zip(a.GetSpacing(), b.GetSpacing(), strict=True):
        if abs(float(av) - float(bv)) > tolerance:
            return False
    for av, bv in zip(a.GetOrigin(), b.GetOrigin(), strict=True):
        if abs(float(av) - float(bv)) > tolerance:
            return False
    for av, bv in zip(a.GetDirection(), b.GetDirection(), strict=True):
        if abs(float(av) - float(bv)) > tolerance:
            return False
    return True


def resample_binary_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """Nearest-neighbor resample a binary mask into the reference geometry."""
    image = sitk.Cast(image > 0, sitk.sitkUInt8)
    if geometry_matches(image, reference):
        return image

    logging.info(
        "Resampling aorta mask to coronary geometry for interface localization: "
        "aorta size=%s spacing=%s; coronary size=%s spacing=%s",
        image.GetSize(),
        image.GetSpacing(),
        reference.GetSize(),
        reference.GetSpacing(),
    )
    return sitk.Cast(
        sitk.Resample(
            image,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        ) > 0,
        sitk.sitkUInt8,
    )


def binary_boundary_image(binary_mask: sitk.Image) -> sitk.Image:
    """Return the fully-connected foreground boundary of a binary mask."""
    contour = sitk.BinaryContourImageFilter()
    contour.SetForegroundValue(1)
    contour.SetBackgroundValue(0)
    contour.SetFullyConnected(True)
    return sitk.Cast(
        contour.Execute(sitk.Cast(binary_mask > 0, sitk.sitkUInt8)),
        sitk.sitkUInt8,
    )


def physical_distance_to_binary_mask(binary_mask: sitk.Image) -> sitk.Image:
    """Physical distance in millimetres to a binary foreground mask."""
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetInsideIsPositive(False)
    distance_filter.SetSquaredDistance(False)
    distance_filter.SetUseImageSpacing(True)
    return distance_filter.Execute(sitk.Cast(binary_mask > 0, sitk.sitkUInt8))


def centered_aorta_coronary_interface_inlet(
    coronary_binary: sitk.Image,
    aorta_binary: sitk.Image,
    coronary_surface: vtk.vtkPolyData,
    aorta_surface: vtk.vtkPolyData,
    *,
    interface_band_mm: float,
    minimum_interface_voxels: int,
    maximum_interface_band_mm: float,
) -> InletProximity:
    """
    Locate the ostium at the center of the coronary-aorta interface patch.

    This intentionally does not use a single globally closest aorta/coronary
    surface pair. Instead it:

    1. extracts the coronary boundary in the segmentation domain;
    2. computes physical distance from every coronary-boundary voxel to aorta;
    3. keeps a narrow distance band around the nearest aortic interface;
    4. separates that band into 26-connected patches;
    5. chooses the largest patch that remains in the nearest-distance tier;
    6. uses the physical centroid of that patch as the ostium-center estimate;
    7. snaps that centroid to the actual coronary mesh for VMTK seeding.

    Using a connected interface patch makes the result insensitive to which one
    surface vertex happens to be closest on an oblong/irregular aorta.
    """
    validate_nonempty_surface(coronary_surface, "Coronary inlet localization")
    validate_nonempty_surface(aorta_surface, "Aorta inlet localization")

    coronary_binary = sitk.Cast(coronary_binary > 0, sitk.sitkUInt8)
    if count_foreground_voxels(coronary_binary) == 0:
        raise ValueError("Coronary mask is empty during inlet localization.")

    aorta_binary = resample_binary_to_reference(aorta_binary, coronary_binary)
    if count_foreground_voxels(aorta_binary) == 0:
        raise ValueError("Aorta mask is empty after alignment to coronary geometry.")

    boundary = binary_boundary_image(coronary_binary)
    boundary_arr = sitk.GetArrayViewFromImage(boundary) > 0
    if not np.any(boundary_arr):
        raise RuntimeError("Coronary boundary extraction produced no voxels.")

    # Measure distance to the AORTA BOUNDARY rather than to the aorta volume.
    # This remains stable when iterative coronary dilation overlaps the aorta:
    # points deeper inside the aorta no longer all collapse to zero distance.
    aorta_boundary = binary_boundary_image(aorta_binary)
    distance_map = physical_distance_to_binary_mask(aorta_boundary)
    distance_arr = np.asarray(sitk.GetArrayViewFromImage(distance_map), dtype=np.float64)
    distance_arr = np.maximum(distance_arr, 0.0)
    boundary_distances = distance_arr[boundary_arr]
    minimum_distance_mm = float(np.min(boundary_distances))

    band_mm = float(interface_band_mm)
    selected_label = None
    selected_stats = None
    selected_component_image = None

    while True:
        threshold_mm = minimum_distance_mm + band_mm
        candidate_arr = boundary_arr & (distance_arr <= threshold_mm)
        candidate = sitk.GetImageFromArray(candidate_arr.astype(np.uint8))
        candidate.CopyInformation(coronary_binary)

        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        components = cc_filter.Execute(candidate)
        shape = sitk.LabelShapeStatisticsImageFilter()
        shape.Execute(components)
        labels = [int(label) for label in shape.GetLabels()]
        if not labels:
            raise RuntimeError("No coronary-aorta interface patch could be identified.")

        component_arr = sitk.GetArrayViewFromImage(components)
        patch_stats: list[tuple[int, int, float, float]] = []
        for label in labels:
            mask = component_arr == label
            values = distance_arr[mask]
            if values.size == 0:
                continue
            patch_stats.append(
                (
                    label,
                    int(shape.GetNumberOfPixels(label)),
                    float(np.min(values)),
                    float(np.median(values)),
                )
            )

        if not patch_stats:
            raise RuntimeError("Interface connected-component analysis returned no usable patch.")

        # Permit multiple patches in the immediate nearest-distance tier, then
        # choose the largest one. This rejects a one-vertex accidental closest
        # contact while retaining the true ostial face when it is broader.
        nearest_tier_slack_mm = max(0.25, min(0.50, 0.5 * band_mm))
        eligible = [
            item
            for item in patch_stats
            if item[2] <= minimum_distance_mm + nearest_tier_slack_mm
        ]
        if not eligible:
            eligible = patch_stats
        selected_stats = max(eligible, key=lambda item: (item[1], -item[3]))
        selected_label = int(selected_stats[0])
        selected_component_image = components

        if (
            selected_stats[1] >= minimum_interface_voxels
            or band_mm >= maximum_interface_band_mm - 1.0e-9
        ):
            break

        band_mm = min(maximum_interface_band_mm, band_mm + 0.25)

    assert selected_label is not None
    assert selected_stats is not None
    assert selected_component_image is not None

    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(selected_component_image)
    centroid = tuple(
        float(value) for value in shape.GetCentroid(selected_label)
    )

    inlet_guess = InletProximity(
        coronary_position=centroid,
        aorta_position=centroid,
        distance_mm=minimum_distance_mm,
        coronary_point_id=-1,
    )
    inlet = snap_inlet_to_coronary_vertex(
        inlet_guess,
        coronary_surface,
        aorta_surface,
    )

    logging.info(
        "Centered ostium interface: min boundary-to-aorta=%.4f mm; "
        "band=%.3f mm; patch=%d voxels; patch min/median=%.4f/%.4f mm; "
        "centroid=[%.4f, %.4f, %.4f]; snapped inlet=[%.4f, %.4f, %.4f]; "
        "snapped inlet-to-aorta=%.4f mm",
        minimum_distance_mm,
        band_mm,
        selected_stats[1],
        selected_stats[2],
        selected_stats[3],
        centroid[0], centroid[1], centroid[2],
        inlet.coronary_position[0], inlet.coronary_position[1], inlet.coronary_position[2],
        inlet.distance_mm,
    )
    if selected_stats[1] < minimum_interface_voxels:
        logging.warning(
            "Centered ostium interface used only %d voxel(s), below the preferred "
            "minimum of %d, after expanding to %.3f mm.",
            selected_stats[1],
            minimum_interface_voxels,
            band_mm,
        )
    return inlet


def snap_inlet_to_coronary_vertex(
    inlet: InletProximity,
    coronary_surface: vtk.vtkPolyData,
    aorta_surface: vtk.vtkPolyData,
) -> InletProximity:
    """Snap a centered interface estimate to the exact coronary vertex used by VMTK."""
    coronary_locator = vtk.vtkPointLocator()
    coronary_locator.SetDataSet(coronary_surface)
    coronary_locator.BuildLocator()
    coronary_point_id = coronary_locator.FindClosestPoint(
        inlet.coronary_position
    )
    if coronary_point_id < 0:
        raise RuntimeError("Failed to snap the inlet to a coronary mesh vertex.")

    coronary_position = tuple(
        float(value) for value in coronary_surface.GetPoint(coronary_point_id)
    )

    aorta_locator = vtk.vtkCellLocator()
    aorta_locator.SetDataSet(aorta_surface)
    aorta_locator.BuildLocator()
    closest_aorta_point = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    distance_squared = vtk.reference(0.0)
    aorta_locator.FindClosestPoint(
        coronary_position,
        closest_aorta_point,
        cell_id,
        sub_id,
        distance_squared,
    )

    return InletProximity(
        coronary_position=coronary_position,
        aorta_position=tuple(float(value) for value in closest_aorta_point),
        distance_mm=float(np.sqrt(float(distance_squared))),
        coronary_point_id=int(coronary_point_id),
    )

def read_polydata(path: Path) -> vtk.vtkPolyData:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        reader: Any = vtk.vtkSTLReader()
    elif suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif suffix == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif suffix == ".ply":
        reader = vtk.vtkPLYReader()
    elif suffix == ".obj":
        reader = vtk.vtkOBJReader()
    else:
        raise ValueError(
            f"Unsupported mesh format {suffix!r}. Supported: STL, VTP, VTK, "
            "PLY, OBJ."
        )

    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(reader.GetOutput())
    validate_nonempty_surface(output, f"Reading {path.name}")
    return output


def write_polydata_vtp(polydata: vtk.vtkPolyData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise IOError(f"VTK failed to write: {path}")


def extract_centerline_cell(centerline: vtk.vtkPolyData, cell_id: int) -> vtk.vtkPolyData:
    """Extract one VMTK source-to-target centerline cell while preserving arrays."""
    if cell_id < 0 or cell_id >= centerline.GetNumberOfCells():
        raise IndexError(f"Centerline cell id out of range: {cell_id}")

    ids = vtk.vtkIdList()
    ids.InsertNextId(int(cell_id))

    extractor = vtk.vtkExtractCells()
    extractor.SetInputData(centerline)
    extractor.SetCellList(ids)
    extractor.Update()

    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(extractor.GetOutputPort())
    geometry.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(geometry.GetOutput())
    if output.GetNumberOfPoints() == 0 or output.GetNumberOfCells() == 0:
        raise RuntimeError(f"Failed to extract centerline cell {cell_id}.")
    return output


def write_individual_centerlines(
    centerline: vtk.vtkPolyData,
    segmentation_path: Path,
) -> list[Path]:
    """Write one numbered VTP file for every cell in the combined VMTK centerline."""
    written: list[Path] = []
    for cell_id in range(centerline.GetNumberOfCells()):
        path = individual_centerline_path(segmentation_path, cell_id + 1)
        write_polydata_vtp(extract_centerline_cell(centerline, cell_id), path)
        written.append(path)
    return written


def write_slicer_endpoint_markups(
    path: Path,
    endpoint_positions: Sequence[Sequence[float]],
) -> None:
    """Write final inlet + outlets as a 3D Slicer Fiducial .mrk.json file."""
    if not endpoint_positions:
        raise ValueError("At least one endpoint is required for markup output.")

    control_points = []
    for index, position in enumerate(endpoint_positions):
        label = "Inlet" if index == 0 else f"Outlet_{index:02d}"
        control_points.append(
            {
                "id": str(index + 1),
                "label": label,
                "description": "Centered coronary-aorta inlet" if index == 0 else "Final distal endpoint",
                "associatedNodeID": "",
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }
        )

    payload = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(control_points),
                "controlPoints": control_points,
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 0.4],
                    "selectedColor": [1.0, 0.5, 0.5],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 1.5,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "snapMode": "toVisibleSurface",
                },
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def inlet_pair_polydata(inlet: InletProximity) -> vtk.vtkPolyData:
    """Create a two-point line showing the selected coronary-aorta gap."""
    points = vtk.vtkPoints()
    points.InsertNextPoint(inlet.coronary_position)
    points.InsertNextPoint(inlet.aorta_position)

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(line)

    output = vtk.vtkPolyData()
    output.SetPoints(points)
    output.SetLines(lines)

    role = vtk.vtkStringArray()
    role.SetName("Role")
    role.InsertNextValue("CoronaryInlet")
    role.InsertNextValue("ClosestAortaPoint")
    output.GetPointData().AddArray(role)

    distance = vtk.vtkDoubleArray()
    distance.SetName("AortaDistanceMm")
    distance.SetNumberOfTuples(2)
    distance.SetValue(0, inlet.distance_mm)
    distance.SetValue(1, inlet.distance_mm)
    output.GetPointData().AddArray(distance)
    return output


def preprocess_for_endpoint_detection(
    surface: vtk.vtkPolyData,
    target_points: int,
) -> vtk.vtkPolyData:
    """
    Standalone replacement for the Slicer module's preprocessing stage.

    The Slicer code invokes its FastQuadric CLI. Here, vtkDecimatePro with
    topology preservation is used to avoid a Slicer dependency and to reduce
    the risk of deleting coronary branches.
    """
    processed = triangulate_and_clean(surface)
    number_of_points = processed.GetNumberOfPoints()

    if target_points > 0 and number_of_points > target_points:
        retained_fraction = target_points / float(number_of_points)
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(processed)
        decimator.SetTargetReduction(1.0 - retained_fraction)
        decimator.PreserveTopologyOn()
        decimator.SplittingOff()
        decimator.BoundaryVertexDeletionOff()
        decimator.Update()

        decimated = vtk.vtkPolyData()
        decimated.DeepCopy(decimator.GetOutput())
        validate_nonempty_surface(decimated, "Endpoint preprocessing")
        processed = triangulate_and_clean(decimated)

    return recompute_normals(processed)


def open_surface_at_point(
    polydata: vtk.vtkPolyData,
    hole_position: Sequence[float],
) -> None:
    """Delete one triangle nearest a requested point, matching Slicer logic."""
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(polydata)
    point_locator.BuildLocator()
    hole_point_index = point_locator.FindClosestPoint(hole_position)
    if hole_point_index < 0:
        raise ValueError("Cannot open an empty surface.")

    polydata.BuildLinks()
    cell_ids = vtk.vtkIdList()
    polydata.GetPointCells(hole_point_index, cell_ids)
    if cell_ids.GetNumberOfIds() == 0:
        raise RuntimeError("No surface cell was found at the requested hole.")

    polydata.DeleteCell(cell_ids.GetId(0))
    polydata.RemoveDeletedCells()


def extract_network(
    surface: vtk.vtkPolyData,
    vmtk_modules: SimpleNamespace,
    preferred_start_position: Sequence[float] | None,
) -> vtk.vtkPolyData:
    simplified = triangulate_and_clean(surface)
    opened_surface = vtk.vtkPolyData()
    opened_surface.DeepCopy(simplified)

    if preferred_start_position is None:
        bounds = opened_surface.GetBounds()
        hole_position = (bounds[0], bounds[2], bounds[4])
    else:
        hole_position = preferred_start_position
    open_surface_at_point(opened_surface, hole_position)

    network_extraction = (
        vmtk_modules.misc.vtkvmtkPolyDataNetworkExtraction()
    )
    network_extraction.SetInputData(opened_surface)
    network_extraction.SetAdvancementRatio(1.05)
    network_extraction.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    network_extraction.SetTopologyArrayName(TOPOLOGY_ARRAY_NAME)
    network_extraction.SetMarksArrayName(MARKS_ARRAY_NAME)
    network_extraction.Update()

    network = vtk.vtkPolyData()
    network.DeepCopy(network_extraction.GetOutput())
    if network.GetNumberOfPoints() == 0 or network.GetNumberOfCells() == 0:
        raise RuntimeError("VMTK network extraction produced an empty network.")
    return network


def get_network_endpoints(
    network_polydata: vtk.vtkPolyData,
    inlet_anchor_position: Sequence[float],
) -> list[tuple[float, float, float]]:
    """
    Return graph endpoints with the aorta-adjacent inlet endpoint first.

    Endpoint topology is detected as in ExtractCenterlineLogic.getEndPoints,
    but inlet ordering is based only on proximity to the aorta-derived anchor.
    The prior largest-radius fallback is intentionally not used.
    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(network_polydata)
    cleaner.Update()

    network = vtk.vtkPolyData()
    network.DeepCopy(cleaner.GetOutput())
    network.BuildCells()
    network.BuildLinks(0)

    points = network.GetPoints()
    endpoint_ids = vtk.vtkIdList()
    start_point_id = -1
    minimum_distance_squared = np.inf

    for cell_index in range(network.GetNumberOfCells()):
        cell = network.GetCell(cell_index)
        number_of_cell_points = cell.GetNumberOfPoints()
        if number_of_cell_points < 2:
            continue

        for cell_point_index in (0, number_of_cell_points - 1):
            point_id = cell.GetPointId(cell_point_index)
            point_cells = vtk.vtkIdList()
            network.GetPointCells(point_id, point_cells)
            if point_cells.GetNumberOfIds() != 1:
                continue

            endpoint_ids.InsertUniqueId(point_id)
            position = points.GetPoint(point_id)

            distance_squared = vtk.vtkMath.Distance2BetweenPoints(
                position,
                inlet_anchor_position,
            )
            if distance_squared < minimum_distance_squared:
                minimum_distance_squared = distance_squared
                start_point_id = point_id

    if endpoint_ids.GetNumberOfIds() == 0:
        return []
    if start_point_id < 0:
        start_point_id = endpoint_ids.GetId(0)

    endpoint_positions = [tuple(float(v) for v in points.GetPoint(start_point_id))]
    for endpoint_index in range(endpoint_ids.GetNumberOfIds()):
        point_id = endpoint_ids.GetId(endpoint_index)
        if point_id == start_point_id:
            continue
        endpoint_positions.append(
            tuple(float(v) for v in points.GetPoint(point_id))
        )
    return endpoint_positions


def auto_detect_endpoints(
    surface: vtk.vtkPolyData,
    inlet: InletProximity,
    vmtk_modules: SimpleNamespace,
    target_points: int,
    dilation_iteration: int,
    foreground_voxels: int,
) -> EndpointDetection:
    preprocessed = preprocess_for_endpoint_detection(surface, target_points)
    network = extract_network(
        preprocessed,
        vmtk_modules,
        preferred_start_position=inlet.coronary_position,
    )
    endpoints = get_network_endpoints(
        network,
        inlet_anchor_position=inlet.coronary_position,
    )
    if len(endpoints) < 2:
        raise RuntimeError(
            f"Automatic endpoint detection found only {len(endpoints)} "
            "endpoint(s); at least two are required."
        )

    return EndpointDetection(
        dilation_iteration=dilation_iteration,
        endpoint_positions=endpoints,
        inlet_coronary_position=inlet.coronary_position,
        inlet_aorta_position=inlet.aorta_position,
        inlet_distance_mm=inlet.distance_mm,
        network=network,
        surface=surface,
        foreground_voxels=foreground_voxels,
    )


def history_record_from_detection(
    detection: EndpointDetection,
    unchanged_from_previous: bool,
    stable_comparison_run: int,
) -> HistoryRecord:
    source = (
        detection.endpoint_positions[0]
        if detection.endpoint_positions
        else (None, None, None)
    )
    return HistoryRecord(
        dilation_iteration=detection.dilation_iteration,
        endpoint_count=detection.endpoint_count,
        unchanged_from_previous=unchanged_from_previous,
        stable_comparison_run=stable_comparison_run,
        foreground_voxels=detection.foreground_voxels,
        mesh_points=detection.surface.GetNumberOfPoints(),
        mesh_triangles=detection.surface.GetNumberOfPolys(),
        source_x=source[0],
        source_y=source[1],
        source_z=source[2],
        inlet_anchor_x=detection.inlet_coronary_position[0],
        inlet_anchor_y=detection.inlet_coronary_position[1],
        inlet_anchor_z=detection.inlet_coronary_position[2],
        aorta_x=detection.inlet_aorta_position[0],
        aorta_y=detection.inlet_aorta_position[1],
        aorta_z=detection.inlet_aorta_position[2],
        aorta_distance_mm=detection.inlet_distance_mm,
    )


def find_stable_endpoints(
    segmentation_path: Path,
    original_surface: vtk.vtkPolyData,
    aorta_surface: vtk.vtkPolyData,
    aorta_binary: sitk.Image,
    binary_segmentation: sitk.Image,
    original_inlet: InletProximity,
    config: Configuration,
    vmtk_modules: SimpleNamespace,
) -> tuple[EndpointDetection, list[HistoryRecord]]:
    """Find the first endpoint set whose count is stable versus the prior pass."""
    history: list[HistoryRecord] = []

    baseline = auto_detect_endpoints(
        original_surface,
        original_inlet,
        vmtk_modules,
        target_points=config.endpoint_target_points,
        dilation_iteration=0,
        foreground_voxels=count_foreground_voxels(binary_segmentation),
    )
    history.append(
        history_record_from_detection(
            baseline,
            unchanged_from_previous=False,
            stable_comparison_run=0,
        )
    )
    logging.info(
        "Dilation 0: %d endpoints; aorta distance=%.4f mm "
        "(initial provided/generated mesh)",
        baseline.endpoint_count,
        baseline.inlet_distance_mm,
    )

    previous_detection = baseline
    current_binary = binary_segmentation
    stable_run = 0

    for dilation_iteration in range(1, config.max_dilations + 1):
        current_binary = dilate_binary_image(
            current_binary,
            config.dilation_radius,
            config.dilation_kernel,
        )
        dilated_surface = binary_image_to_surface(current_binary, config)
        current_inlet = centered_aorta_coronary_interface_inlet(
            current_binary,
            aorta_binary,
            dilated_surface,
            aorta_surface,
            interface_band_mm=config.inlet_interface_band_mm,
            minimum_interface_voxels=config.inlet_interface_min_voxels,
            maximum_interface_band_mm=config.inlet_interface_max_band_mm,
        )
        current_detection = auto_detect_endpoints(
            dilated_surface,
            current_inlet,
            vmtk_modules,
            target_points=config.endpoint_target_points,
            dilation_iteration=dilation_iteration,
            foreground_voxels=count_foreground_voxels(current_binary),
        )

        unchanged = (
            current_detection.endpoint_count
            == previous_detection.endpoint_count
        )
        stable_run = stable_run + 1 if unchanged else 0
        history.append(
            history_record_from_detection(
                current_detection,
                unchanged_from_previous=unchanged,
                stable_comparison_run=stable_run,
            )
        )
        logging.info(
            "Dilation %d: %d endpoints; aorta distance=%.4f mm; "
            "unchanged=%s; stable run=%d/%d",
            dilation_iteration,
            current_detection.endpoint_count,
            current_detection.inlet_distance_mm,
            unchanged,
            stable_run,
            config.stable_comparisons,
        )


        if stable_run >= config.stable_comparisons:
            return current_detection, history

        previous_detection = current_detection

    counts = [record.endpoint_count for record in history]
    raise EndpointStabilityError(
        "Endpoint count did not stabilize within "
        f"{config.max_dilations} dilation(s). Counts were: {counts}",
        history,
    )


def snap_points_to_surface(
    positions: Sequence[Sequence[float]],
    surface: vtk.vtkPolyData,
) -> list[tuple[float, float, float]]:
    """Snap detected network endpoints to vertices of the original surface."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    snapped: list[tuple[float, float, float]] = []
    for position in positions:
        point_id = locator.FindClosestPoint(position)
        if point_id < 0:
            raise RuntimeError("Failed to snap an endpoint to the original mesh.")
        snapped.append(tuple(float(v) for v in surface.GetPoint(point_id)))
    return snapped


def extract_centerline(
    original_surface: vtk.vtkPolyData,
    endpoint_positions: Sequence[Sequence[float]],
    curve_sampling_distance: float,
    simplify_voronoi: bool,
    vmtk_modules: SimpleNamespace,
) -> tuple[vtk.vtkPolyData, vtk.vtkPolyData, list[int]]:
    """Compute a source-to-all-target VMTK centerline on the original mesh."""
    if len(endpoint_positions) < 2:
        raise ValueError("At least two endpoints are required.")

    capper = vmtk_modules.computational.vtkvmtkCapPolyData()
    capper.SetInputData(original_surface)
    capper.SetDisplacement(0.0)
    capper.SetInPlaneDisplacement(0.0)
    capper.Update()

    capped_surface = vtk.vtkPolyData()
    capped_surface.DeepCopy(capper.GetOutput())
    validate_nonempty_surface(capped_surface, "VMTK surface capping")

    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(capped_surface)
    point_locator.BuildLocator()

    seed_ids = [point_locator.FindClosestPoint(point) for point in endpoint_positions]
    if any(point_id < 0 for point_id in seed_ids):
        raise RuntimeError("At least one endpoint could not be placed on the surface.")
    if len(set(seed_ids)) != len(seed_ids):
        raise RuntimeError(
            "Two or more endpoints mapped to the same surface seed. This "
            "usually indicates that the accepted dilation was too large or "
            "that a branch endpoint is not spatially distinct."
        )

    source_ids = vtk.vtkIdList()
    source_ids.InsertNextId(seed_ids[0])
    target_ids = vtk.vtkIdList()
    for point_id in seed_ids[1:]:
        target_ids.InsertNextId(point_id)

    centerline_filter = (
        vmtk_modules.computational.vtkvmtkPolyDataCenterlines()
    )
    centerline_filter.SetInputData(capped_surface)
    centerline_filter.SetSourceSeedIds(source_ids)
    centerline_filter.SetTargetSeedIds(target_ids)
    centerline_filter.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    centerline_filter.SetCostFunction("1/R")
    centerline_filter.SetFlipNormals(False)
    centerline_filter.SetAppendEndPointsToCenterlines(0)
    centerline_filter.SetSimplifyVoronoi(bool(simplify_voronoi))
    centerline_filter.SetCenterlineResampling(0)
    centerline_filter.SetResamplingStepLength(curve_sampling_distance)
    centerline_filter.Update()

    output = centerline_filter.GetOutput()
    if output is None or output.GetNumberOfPoints() == 0:
        raise RuntimeError("VMTK centerline extraction generated no output.")
    centerline = vtk.vtkPolyData()
    centerline.DeepCopy(output)

    voronoi_output = centerline_filter.GetVoronoiDiagram()
    if voronoi_output is None or voronoi_output.GetNumberOfPoints() == 0:
        raise RuntimeError("VMTK generated no Voronoi diagram.")
    voronoi = vtk.vtkPolyData()
    voronoi.DeepCopy(voronoi_output)

    return centerline, voronoi, seed_ids


def process_segmentation(
    segmentation_path: Path,
    config: Configuration,
    vmtk_modules: SimpleNamespace,
) -> None:
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Segmentation does not exist: {segmentation_path}")

    outputs = output_paths_for(segmentation_path)
    ensure_outputs_available(segmentation_path, outputs, config.overwrite)

    reference_image, binary_segmentation = read_binary_segmentation(
        segmentation_path,
        config.foreground_label,
    )
    (
        aorta_label_image_path,
        _aorta_reference,
        aorta_binary,
    ) = resolve_aorta_label_image(
        segmentation_path,
        reference_image,
        config,
    )

    # Working surfaces exist only in memory. No STL, image, or segmentation copy
    # is written by this script.
    original_surface = binary_image_to_surface(binary_segmentation, config)
    original_surface = recompute_normals(triangulate_and_clean(original_surface))
    aorta_surface = binary_image_to_reference_surface(aorta_binary)

    original_inlet = centered_aorta_coronary_interface_inlet(
        binary_segmentation,
        aorta_binary,
        original_surface,
        aorta_surface,
        interface_band_mm=config.inlet_interface_band_mm,
        minimum_interface_voxels=config.inlet_interface_min_voxels,
        maximum_interface_band_mm=config.inlet_interface_max_band_mm,
    )

    logging.info("Segmentation: %s", segmentation_path)
    logging.info("Aorta label image: %s", aorta_label_image_path)
    logging.info(
        "Image size=%s spacing=%s origin=%s",
        reference_image.GetSize(),
        reference_image.GetSpacing(),
        reference_image.GetOrigin(),
    )

    stable_detection, history = find_stable_endpoints(
        segmentation_path,
        original_surface,
        aorta_surface,
        aorta_binary,
        binary_segmentation,
        original_inlet,
        config,
        vmtk_modules,
    )

    snapped_outlet_positions = snap_points_to_surface(
        stable_detection.endpoint_positions[1:],
        original_surface,
    )
    snapped_positions = [
        original_inlet.coronary_position,
        *snapped_outlet_positions,
    ]

    centerline, _voronoi, _seed_ids = extract_centerline(
        original_surface,
        snapped_positions,
        config.curve_sampling_distance,
        config.simplify_voronoi,
        vmtk_modules,
    )

    # Persist only the requested final artifacts.
    write_polydata_vtp(centerline, outputs.centerline)
    individual_paths = write_individual_centerlines(centerline, segmentation_path)
    write_slicer_endpoint_markups(outputs.endpoints_markup, snapped_positions)

    print(f"  Segmentation:       {segmentation_path}")
    print(f"  Aorta label image:  {aorta_label_image_path}")
    print(
        "  Endpoint counts:    "
        + " -> ".join(str(record.endpoint_count) for record in history)
    )
    print(
        f"  Stable endpoints:   {stable_detection.endpoint_count} "
        f"at dilation {stable_detection.dilation_iteration}"
    )
    print(
        f"  Centered inlet gap: {original_inlet.distance_mm:.4f} mm "
        "(interface-centroid seed to aorta surface)"
    )
    print(
        f"  Combined centerline:{centerline.GetNumberOfPoints():,} points, "
        f"{centerline.GetNumberOfCells():,} cells"
    )
    print(f"  Output centerline:  {outputs.centerline.name}")
    print(f"  Individual paths:   {len(individual_paths)} VTP file(s)")
    print(f"  Endpoints markup:   {outputs.endpoints_markup.name}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_configuration(argv)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        vmtk_modules = load_vmtk_modules()
        segmentation_paths = resolve_segmentation_paths(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not segmentation_paths:
        print("ERROR: No input segmentations were found.", file=sys.stderr)
        return 1

    succeeded = 0
    skipped_existing = 0
    failed = 0

    print(f"Found {len(segmentation_paths)} segmentation(s).")
    print(f"Processed coronary root: {config.input_root}")
    print(f"Raw multi-label root: {config.aorta_root}")
    print(f"Aorta label: {config.aorta_label}")
    print(
        "Dilation step: radius "
        f"{config.dilation_radius}, kernel={config.dilation_kernel}"
    )
    print(
        f"Stability rule: {config.stable_comparisons} unchanged consecutive "
        "comparison(s)"
    )
    print()

    for segmentation_path in segmentation_paths:
        print(f"[{segmentation_path.parent.name} / {segmentation_path.name}]")
        try:
            process_segmentation(segmentation_path, config, vmtk_modules)
            succeeded += 1
        except FileExistsError as exc:
            print(f"  SKIPPED: {exc}", file=sys.stderr)
            skipped_existing += 1
        except Exception as exc:
            logging.exception("Failed to process %s", segmentation_path)
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed += 1
        print()

    print(
        f"Finished: {succeeded} succeeded, {skipped_existing} skipped, "
        f"{failed} failed."
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

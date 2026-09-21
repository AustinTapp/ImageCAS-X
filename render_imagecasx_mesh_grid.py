import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import SimpleITK as sitk
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


def natural_key(path):
    import re
    # Accept either pathlib.Path objects or plain strings.
    name = Path(str(path)).name
    if name.lower().endswith('.nii.gz'):
        name = name[:-7]
    else:
        name = Path(name).stem
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_segmentation(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing_xyz = img.GetSpacing()  # x,y,z
    origin_xyz = img.GetOrigin()
    direction = img.GetDirection()
    return img, arr, spacing_xyz, origin_xyz, direction


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
    relabel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    arr = sitk.GetArrayFromImage(relabel)
    return arr == 1


def extract_surface(mask_zyx: np.ndarray, spacing_xyz: Tuple[float, float, float],
                    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                    direction: Optional[Tuple[float, ...]] = None,
                    keep_lcc: bool = False, smooth_iters: int = 30,
                    decimate: Optional[float] = None) -> pv.PolyData:
    mask = mask_zyx.astype(bool)
    if keep_lcc and mask.any():
        mask = largest_connected_component(mask)

    # SimpleITK arrays are z,y,x. PyVista ImageData expects x,y,z.
    vol_xyz = np.transpose(mask.astype(np.uint8), (2, 1, 0))
    nx, ny, nz = vol_xyz.shape

    # Contour() requires POINT data, not CELL data.  ImageData dimensions
    # therefore match the number of voxel samples directly.
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    grid.spacing = tuple(float(v) for v in spacing_xyz)
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data['values'] = vol_xyz.flatten(order='F')

    surf = grid.contour(isosurfaces=[0.5], scalars='values')
    if surf.n_points == 0:
        return surf

    surf = surf.triangulate().clean()

    # PyVista ImageData handles spacing/origin but not arbitrary SimpleITK
    # direction cosines.  Transform the vertices into the NIfTI/SimpleITK
    # physical coordinate system explicitly so exported meshes align with
    # the source segmentation.
    if direction is not None and len(direction) == 9:
        d = np.asarray(direction, dtype=float).reshape(3, 3)
        pts = np.asarray(surf.points, dtype=float)
        pts = pts @ d.T
        pts += np.asarray(origin_xyz, dtype=float)
        surf.points = pts
    else:
        surf.points = np.asarray(surf.points) + np.asarray(origin_xyz, dtype=float)

    if smooth_iters and smooth_iters > 0:
        surf = surf.smooth(
            n_iter=smooth_iters,
            feature_smoothing=False,
            boundary_smoothing=True,
        )
    if decimate is not None and 0.0 < decimate < 1.0 and surf.n_cells > 1000:
        surf = surf.decimate(decimate)
    surf = surf.compute_normals(auto_orient_normals=True, inplace=False)
    return surf


def save_mesh(mesh: pv.PolyData, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == '.stl':
        mesh.save(str(out_path))
    elif suffix == '.vtp':
        mesh.save(str(out_path))
    elif suffix == '.ply':
        mesh.save(str(out_path))
    else:
        mesh.save(str(out_path.with_suffix('.vtp')))


def choose_camera(mesh: pv.PolyData):
    bounds = mesh.bounds  # xmin,xmax,ymin,ymax,zmin,zmax
    center = np.array(mesh.center)
    lengths = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]], dtype=float)
    max_len = max(lengths.max(), 1.0)
    # oblique view generally good for tubular anatomy
    direction = np.array([1.6, -1.2, 0.9], dtype=float)
    direction = direction / np.linalg.norm(direction)
    position = center + direction * (3.0 * max_len)
    viewup = (0.0, 0.0, 1.0)
    return position.tolist(), center.tolist(), viewup


def render_mesh(mesh: pv.PolyData, out_png: Path, size=(500, 500), color='lightcoral',
                background='white'):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(background)
    if mesh.n_points > 0:
        pl.add_mesh(mesh, color=color, smooth_shading=True, specular=0.15)
        pl.add_axes(line_width=1, labels_off=True)
        pl.camera_position = choose_camera(mesh)
        pl.camera.zoom(1.2)
    pl.show(screenshot=str(out_png), auto_close=True)


def get_font(size: int, bold: bool = False):
    """Load a readable TrueType font, with Windows-friendly fallbacks."""
    candidates = []
    if os.name == 'nt':
        candidates.extend([
            r'C:\\Windows\\Fonts\\arialbd.ttf' if bold else r'C:\\Windows\\Fonts\\arial.ttf',
            r'C:\\Windows\\Fonts\\calibrib.ttf' if bold else r'C:\\Windows\\Fonts\\calibri.ttf',
        ])
    candidates.extend([
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ])
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def annotate_image(img_path: Path, header: str = '', footer: str = '', font_size: int = 28) -> Image.Image:
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    pad = font_size + 18
    top_pad = pad if header else 0
    bottom_pad = pad if footer else 0
    canvas = Image.new('RGB', (w, h + top_pad + bottom_pad), 'white')
    canvas.paste(img, (0, top_pad))
    draw = ImageDraw.Draw(canvas)
    font = get_font(font_size, bold=True)
    if header:
        bbox = draw.textbbox((0, 0), header, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, (top_pad - th) // 2), header, fill='black', font=font)
    if footer:
        bbox = draw.textbbox((0, 0), footer, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, h + top_pad + (bottom_pad - th) // 2), footer, fill='black', font=font)
    return canvas


def build_montage(render_map: Dict[str, Dict[str, Path]], cases: List[str], models: List[str],
                  out_path: Path, cell_size=(500, 592), label_col_width=240, font_size=30):
    cell_w, cell_h = cell_size
    header_h = font_size + 30
    rows = len(cases)
    cols = len(models)
    canvas_w = label_col_width + cols * cell_w
    canvas_h = header_h + rows * cell_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = get_font(font_size, bold=True)

    # headers
    for j, model in enumerate(models):
        x = label_col_width + j * cell_w
        draw.rectangle([x, 0, x + cell_w, header_h], outline='black', width=1)
        bbox = draw.textbbox((0, 0), model, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (cell_w - tw)//2, (header_h - th)//2), model, fill='black', font=font)

    draw.rectangle([0, 0, label_col_width, header_h], outline='black', width=1)
    bbox = draw.textbbox((0, 0), 'Case / Model', font=font)
    th = bbox[3] - bbox[1]
    draw.text((12, (header_h - th)//2), 'Case / Model', fill='black', font=font)

    for i, case in enumerate(cases):
        y = header_h + i * cell_h
        draw.rectangle([0, y, label_col_width, y + cell_h], outline='black', width=1)
        bbox = draw.textbbox((0, 0), case, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((12, y + (cell_h - th)//2), case, fill='black', font=font)

        for j, model in enumerate(models):
            x = label_col_width + j * cell_w
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline='black', width=1)
            img_path = render_map.get(case, {}).get(model)
            if img_path is not None and img_path.exists():
                img = Image.open(img_path).convert('RGB')
                img = img.resize((cell_w, cell_h))
                canvas.paste(img, (x, y))
            else:
                missing = 'missing'
                bbox = draw.textbbox((0, 0), missing, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((x + (cell_w - tw)//2, y + (cell_h - th)//2), missing, fill='red', font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def discover_predictions(results_root: Path):
    model_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    ignore = {'meshes', 'renders', 'grids'}
    model_dirs = [p for p in model_dirs if p.name not in ignore]
    model_to_cases: Dict[str, Dict[str, Path]] = {}
    all_cases = set()
    for model_dir in sorted(model_dirs):
        case_files = {}
        nii_files = sorted(list(model_dir.glob('*.nii.gz')) + list(model_dir.glob('*.nii')),
                           key=natural_key)
        for f in nii_files:
            case_name = f.name.replace('.nii.gz', '').replace('.nii', '')
            case_files[case_name] = f
            all_cases.add(case_name)
        if case_files:
            model_to_cases[model_dir.name] = case_files
    return model_to_cases, sorted(all_cases, key=natural_key)


def main():
    parser = argparse.ArgumentParser(description='Render ImageCAS-X predictions as meshes and build a comparison grid.')
    parser.add_argument('--results-root', type=str, default=r'D:\Code\ImageCAS-X\image_results')
    parser.add_argument('--mesh-root', type=str, default=None,
                        help='Default: <results-root>\\meshes')
    parser.add_argument('--render-root', type=str, default=None,
                        help='Default: <results-root>\\renders')
    parser.add_argument('--grid-out', type=str, default=None,
                        help='Default: <results-root>\\grids\\mesh_comparison_grid.png')
    parser.add_argument('--mesh-format', type=str, default='stl', choices=['stl', 'vtp', 'ply'])
    parser.add_argument('--labels', type=int, nargs='*', default=None,
                        help='If provided, render only these labels. Default: all voxels > 0.')
    parser.add_argument('--keep-largest-component', action='store_true')
    parser.add_argument('--smooth-iters', type=int, default=30)
    parser.add_argument('--decimate', type=float, default=None,
                        help='Optional mesh decimation fraction between 0 and 1, e.g. 0.5')
    parser.add_argument('--image-size', type=int, nargs=2, default=[500, 500], metavar=('W', 'H'))
    parser.add_argument('--font-size', type=int, default=30,
                        help='Font size for model/case labels in the comparison graphic. Default: 30')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    results_root = Path(args.results_root)
    mesh_root = Path(args.mesh_root) if args.mesh_root else results_root / 'meshes'
    render_root = Path(args.render_root) if args.render_root else results_root / 'renders'
    grid_out = Path(args.grid_out) if args.grid_out else results_root / 'grids' / 'mesh_comparison_grid.png'

    model_to_cases, cases = discover_predictions(results_root)
    if not model_to_cases:
        raise FileNotFoundError(f'No model prediction folders found in {results_root}')
    models = sorted(model_to_cases.keys())

    print('Discovered models:')
    for model in models:
        print(f'  - {model}: {len(model_to_cases[model])} case(s)')
    print(f'Total cases: {len(cases)}')

    render_map: Dict[str, Dict[str, Path]] = {case: {} for case in cases}

    for model in models:
        for case, pred_path in sorted(model_to_cases[model].items(), key=lambda x: natural_key(Path(x[0]))):
            mesh_out = mesh_root / model / f'{case}.{args.mesh_format}'
            render_out = render_root / model / f'{case}.png'

            if render_out.exists() and mesh_out.exists() and not args.overwrite:
                render_map[case][model] = render_out
                print(f'[skip] {model} / {case}')
                continue

            print(f'[proc] {model} / {case}')
            _, arr, spacing_xyz, origin_xyz, direction = load_segmentation(pred_path)
            if args.labels:
                mask = np.isin(arr, args.labels)
            else:
                mask = arr > 0

            if not np.any(mask):
                print(f'  -> empty prediction; no mesh created')
                continue

            mesh = extract_surface(
                mask_zyx=mask,
                spacing_xyz=spacing_xyz,
                origin_xyz=origin_xyz,
                direction=direction,
                keep_lcc=args.keep_largest_component,
                smooth_iters=args.smooth_iters,
                decimate=args.decimate,
            )
            if mesh.n_points == 0:
                print(f'  -> could not generate surface')
                continue

            save_mesh(mesh, mesh_out)
            render_mesh(mesh, render_out, size=tuple(args.image_size))
            annotated = annotate_image(render_out, header=model, footer=case, font_size=args.font_size)
            annotated.save(render_out)
            render_map[case][model] = render_out

    # rebuild case list from actual render map union to preserve original discovery order
    build_montage(
        render_map=render_map,
        cases=cases,
        models=models,
        out_path=grid_out,
        cell_size=(args.image_size[0], args.image_size[1] + 2 * (args.font_size + 18)),
        label_col_width=max(240, args.font_size * 8),
        font_size=args.font_size,
    )

    print('\nDone.')
    print(f'Meshes saved in:   {mesh_root}')
    print(f'Renders saved in:  {render_root}')
    print(f'Grid saved in:     {grid_out}')


if __name__ == '__main__':
    main()

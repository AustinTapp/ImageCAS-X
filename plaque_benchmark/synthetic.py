"""Create an entirely artificial SHK-shaped cohort for smoke tests."""
from pathlib import Path
import nibabel as nib
import numpy as np
from .data import BRANCHES, save_image, write_json


def create(out, cases=9):
    if cases < 3:
        raise ValueError("At least three synthetic patients are required.")
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Choose an empty synthetic output directory.")
    images,masks = out/"AsNifti",out/"MasksUnverified"
    affine = np.array([[0,-.7,0,40],[.6,0,0,-30],[0,0,.9,12],[0,0,0,1.]])
    x,y,z = np.indices((32,32,32))
    seeds = {}
    for case in range(1,cases+1):
        name = f"SHK{case:04d}"
        ct = np.full(x.shape,-1000,np.float32)
        seeds[name] = {}
        for i,branch in enumerate(BRANCHES):
            center = 6+i*9
            lumen = ((x-center)**2+(y-16)**2 < 5) & (z >= 4) & (z <= 27)
            labels = lumen.astype(np.uint8)
            if (case+i) % 2:
                plaque = (x == center+2) & (y >= 15) & (y <= 17) & (z >= 13) & (z <= 16)
                labels[plaque & ~lumen] = 2
            ct[labels == 1],ct[labels == 2] = 350,150
            save_image(masks/name/f"{branch}.nii.gz",labels,affine)
            seeds[name][branch] = nib.affines.apply_affine(affine,[center,16,6]).tolist()
        save_image(images/name/"series_random.nii.gz",ct,affine)
    write_json(out/"seeds_ras.json",seeds)
    return images,masks

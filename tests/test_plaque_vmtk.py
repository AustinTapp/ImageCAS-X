"""Actual VMTK smoke tests. Run inside the VMTK environment."""
import nibabel as nib
import numpy as np
import pytest
from plaque_benchmark.geometry import _sitk_image,extract_branch
from plaque_benchmark import vmtk_aorta_guided_core as core


@pytest.fixture
def modules():
    try:
        return core.load_vmtk_modules()
    except ImportError:
        pytest.skip("VMTK is unavailable in this interpreter.")


def test_lps_ras_and_output(tmp_path,modules):
    import vtk
    x,y,z = np.indices((25,25,52))
    labels = (((x-12)**2+(y-12)**2 <= 9) & (z > 4) & (z < 47)).astype(np.uint8)
    affine = np.diag([-.7,.8,.9,1.])
    affine[:3,3] = [35,-20,12]
    image = _sitk_image(labels,affine)
    index = [12,12,7]
    ras = nib.affines.apply_affine(affine,index)
    assert np.allclose(image.TransformIndexToPhysicalPoint(index),ras*[-1,-1,1])
    result = extract_branch(labels,affine,tmp_path,"LAD",modules,seed_ras=ras)
    assert result["success"] and result["path_qc"][0]["lumen_inside_fraction"] > .95
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(tmp_path/"LAD_centerline.vtp"))
    reader.Update()
    poly = reader.GetOutput()
    assert poly.GetNumberOfPoints() > 0 and poly.GetPointData().GetArray("Radius") is not None
    with np.load(result["paths"][0]) as path:
        assert np.diff(path["arc_mm"]).max() <= .400001 and path["radius_mm"].min() > 0


def test_aorta_guided(tmp_path,modules):
    x,y,z = np.indices((31,31,60))
    labels = (((x-15)**2+(y-15)**2 <= 9) & (z >= 10) & (z < 53)).astype(np.uint8)
    aorta = (x-15)**2+(y-15)**2+(z-4)**2 <= 64
    result = extract_branch(labels,np.eye(4),tmp_path,"RCA",modules,aorta=aorta)
    assert result["success"] and result["orientation"]["method"] == "aorta_interface"
    assert result["dilation_history"] and result["path_qc"][0]["lumen_inside_fraction"] > .9


def test_disconnected_rejected(tmp_path,modules):
    labels = np.zeros((20,20,20),np.uint8)
    labels[2:5,2:5,2:8],labels[12:15,12:15,12:18] = 1,1
    with pytest.raises(ValueError,match="disconnected"):
        extract_branch(labels,np.eye(4),tmp_path,"LAD",modules)


def test_narrow_vessel_target_reach(tmp_path,modules):
    from plaque_benchmark.synthetic import create
    _,masks = create(tmp_path/"synthetic",cases=3)
    image = nib.load(masks/"SHK0001/LAD.nii.gz")
    seed = nib.affines.apply_affine(image.affine,[6,16,6])
    result = extract_branch(np.asanyarray(image.dataobj),image.affine,tmp_path/"out","LAD",modules,seed_ras=seed)
    assert result["surface_retained_fraction"] == 1
    assert all(path["length_mm"] > 15 and path["endpoints_reached"] for path in result["path_qc"] if path["accepted"])

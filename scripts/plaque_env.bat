@echo off
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
if not defined PLAQUE_PYTHON set "PLAQUE_PYTHON=python"
if not defined VMTK_PYTHON set "VMTK_PYTHON=%PLAQUE_PYTHON%"
if not defined SHK_IMAGES set "SHK_IMAGES=D:\Data\PlaqueDataset\AsNifti"
if not defined SHK_MASKS set "SHK_MASKS=D:\Data\PlaqueDataset\MasksUnverified"
if not defined PLAQUE_BENCH set "PLAQUE_BENCH=experiments\plaque_benchmark"
set "PLAQUE_DATASET=%PLAQUE_BENCH%\prepared\dataset.json"
exit /b 0

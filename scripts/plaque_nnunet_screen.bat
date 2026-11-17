@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
if not defined nnUNet_n_proc_DA set "nnUNet_n_proc_DA=8"
nnUNetv2_train 701 3d_fullres 0 -tr nnUNetTrainerSHK25 -p nnUNetPlans_20G
if errorlevel 1 exit /b 1
nnUNetv2_train 701 3d_fullres 0 -tr nnUNetTrainerSHK25 -p nnUNetResEncUNetLPlans_20G
exit /b %errorlevel%

@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
if not defined nnUNet_raw exit /b 2
if not defined nnUNet_preprocessed exit /b 2
"%PLAQUE_PYTHON%" -m plaque_benchmark nnunet-export --dataset "%PLAQUE_DATASET%" --raw "%nnUNet_raw%" --name Dataset701_SHKPlaque
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark nnunet-install-trainers
if errorlevel 1 exit /b 1
nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity -np 8 -npfp 8 -c 3d_fullres -gpu_memory_target 20 -overwrite_plans_name nnUNetPlans_20G
if errorlevel 1 exit /b 1
nnUNetv2_plan_and_preprocess -d 701 -pl nnUNetPlannerResEncL --verify_dataset_integrity -np 8 -npfp 8 -c 3d_fullres -gpu_memory_target 20 -overwrite_plans_name nnUNetResEncUNetLPlans_20G
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark nnunet-split --exported "%nnUNet_raw%\Dataset701_SHKPlaque" --preprocessed "%nnUNet_preprocessed%"
exit /b %errorlevel%

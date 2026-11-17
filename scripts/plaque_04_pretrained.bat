@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
if not defined MOOSAVI_MODEL (
  echo Set MOOSAVI_MODEL to the released nnUNetTrainer__nnUNetPlans__3d_fullres directory.
  exit /b 2
)
if not defined MOOSAVI_OFFSET (
  echo Set MOOSAVI_OFFSET to the explicitly selected intensity offset.
  exit /b 2
)
"%PLAQUE_PYTHON%" -m plaque_benchmark moosavi-inputs --index "%PLAQUE_BENCH%\smpr\oracle_val\smpr.json" --intensity-offset "%MOOSAVI_OFFSET%" --out "%PLAQUE_BENCH%\moosavi_inputs\oracle_val"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark moosavi-infer --model "%MOOSAVI_MODEL%" --images "%PLAQUE_BENCH%\moosavi_inputs\oracle_val" --out "%PLAQUE_BENCH%\moosavi_scores\oracle_val" %*
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark inverse --dataset "%PLAQUE_DATASET%" --index "%PLAQUE_BENCH%\smpr\oracle_val\smpr.json" --scores "%PLAQUE_BENCH%\moosavi_scores\oracle_val" --out "%PLAQUE_BENCH%\predictions\moosavi_oracle_val" --initialization pretrained
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark evaluate --dataset "%PLAQUE_DATASET%" --predictions "%PLAQUE_BENCH%\predictions\moosavi_oracle_val" --out "%PLAQUE_BENCH%\reports\moosavi_oracle_val"
exit /b %errorlevel%

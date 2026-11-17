@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
for %%M in (cas_net ffr_unet swin_unetr) do (
  call :screen %%M
  if errorlevel 1 exit /b 1
)
"%PLAQUE_PYTHON%" -m plaque_benchmark compare --reports "%PLAQUE_BENCH%\reports\cas_net_val" "%PLAQUE_BENCH%\reports\ffr_unet_val" "%PLAQUE_BENCH%\reports\swin_unetr_val" --out "%PLAQUE_BENCH%\comparison"
exit /b %errorlevel%

:screen
"%PLAQUE_PYTHON%" -m plaque_benchmark train --dataset "%PLAQUE_DATASET%" --model %1 --profile screen --out "%PLAQUE_BENCH%\runs\%1"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark infer --dataset "%PLAQUE_DATASET%" --checkpoint "%PLAQUE_BENCH%\runs\%1\best.pt" --out "%PLAQUE_BENCH%\predictions\%1_val"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark evaluate --dataset "%PLAQUE_DATASET%" --predictions "%PLAQUE_BENCH%\predictions\%1_val" --out "%PLAQUE_BENCH%\reports\%1_val"
exit /b %errorlevel%

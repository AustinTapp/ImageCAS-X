@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark ensemble --dataset "%PLAQUE_DATASET%" --predictions "%PLAQUE_BENCH%\predictions\cas_net_val" "%PLAQUE_BENCH%\predictions\ffr_unet_val" "%PLAQUE_BENCH%\predictions\swin_unetr_val" --out "%PLAQUE_BENCH%\predictions\ensemble_val" %*
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark evaluate --dataset "%PLAQUE_DATASET%" --predictions "%PLAQUE_BENCH%\predictions\ensemble_val" --out "%PLAQUE_BENCH%\reports\ensemble_val"
exit /b %errorlevel%

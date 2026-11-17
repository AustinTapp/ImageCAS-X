@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark smpr --dataset "%PLAQUE_DATASET%" --centerlines "%PLAQUE_BENCH%\centerlines\oracle_val\centerlines.json" --out "%PLAQUE_BENCH%\smpr\oracle_val" --include-labels --overlays %*
exit /b %errorlevel%

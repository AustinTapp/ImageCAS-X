@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
"%VMTK_PYTHON%" -m plaque_benchmark centerlines --dataset "%PLAQUE_DATASET%" --source oracle --partition val --out "%PLAQUE_BENCH%\centerlines\oracle_val" %*
exit /b %errorlevel%

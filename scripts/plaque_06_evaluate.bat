@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
if "%~1"=="" (
  echo Supply the standardized predictions directory.
  exit /b 2
)
"%PLAQUE_PYTHON%" -m plaque_benchmark evaluate --dataset "%PLAQUE_DATASET%" --predictions "%~1" --out "%~1\evaluation"
exit /b %errorlevel%

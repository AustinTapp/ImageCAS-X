@echo off
setlocal
call conda activate PlaquePilot
if errorlevel 1 (
  echo Activate Miniconda through Anaconda Prompt, then rerun this file.
  exit /b 1
)
pushd "%~dp0"
python "%~dp0run_PlaquePilot.py" %*
set "pilot_exit=%errorlevel%"
popd
exit /b %pilot_exit%

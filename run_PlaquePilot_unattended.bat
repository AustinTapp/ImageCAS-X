@echo off
setlocal
call conda activate PlaquePilot
if errorlevel 1 exit /b 1
pushd "%~dp0"
python "%~dp0run_PlaquePilot_unattended.py" %*
set "pilot_exit=%errorlevel%"
popd
exit /b %pilot_exit%

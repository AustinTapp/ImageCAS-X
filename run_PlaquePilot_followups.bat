@echo off
setlocal
cd /d "%~dp0"
python "%~dp0run_PlaquePilot_followups.py" %*
exit /b %errorlevel%

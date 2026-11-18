@echo off
setlocal
cd /d "%~dp0"
python "%~dp0review_plaque_cases.py" %*
exit /b %errorlevel%

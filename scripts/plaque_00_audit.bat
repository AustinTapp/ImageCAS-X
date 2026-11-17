@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark audit --images "%SHK_IMAGES%" --masks "%SHK_MASKS%" --out "%PLAQUE_BENCH%\audit.json" %*
exit /b %errorlevel%

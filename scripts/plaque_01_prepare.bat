@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark split --audit "%PLAQUE_BENCH%\audit.json" --out "%PLAQUE_BENCH%\split.json" %*
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark prepare --audit "%PLAQUE_BENCH%\audit.json" --split "%PLAQUE_BENCH%\split.json" --out "%PLAQUE_BENCH%\prepared"
exit /b %errorlevel%

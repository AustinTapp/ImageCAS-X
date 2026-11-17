@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m pytest tests\test_plaque_benchmark.py -q
if errorlevel 1 exit /b 1
"%VMTK_PYTHON%" -m pytest tests\test_plaque_vmtk.py -q
exit /b %errorlevel%

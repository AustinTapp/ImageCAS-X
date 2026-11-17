@echo off
setlocal
call "%~dp0plaque_env.bat"
if errorlevel 1 exit /b 1
if "%~2"=="" (
  echo Supply a promoted checkpoint and its frozen promotion JSON.
  exit /b 2
)
"%PLAQUE_PYTHON%" -m plaque_benchmark infer --dataset "%PLAQUE_DATASET%" --partition test --checkpoint "%~1" --frozen "%~2" --out "%PLAQUE_BENCH%\predictions\promoted_test"
if errorlevel 1 exit /b 1
"%PLAQUE_PYTHON%" -m plaque_benchmark evaluate --dataset "%PLAQUE_DATASET%" --partition test --predictions "%PLAQUE_BENCH%\predictions\promoted_test" --frozen "%~2" --out "%PLAQUE_BENCH%\reports\promoted_test"
exit /b %errorlevel%

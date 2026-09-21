@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if defined REPRO_PYTHON (
  "%REPRO_PYTHON%" -u "%~dp0revision_gpu\bootstrap_gpu.py" 4 %*
) else if exist "%~dp0.venv-reproduce\Scripts\python.exe" (
  "%~dp0.venv-reproduce\Scripts\python.exe" -u "%~dp0revision_gpu\bootstrap_gpu.py" 4 %*
) else (
  python -u "%~dp0revision_gpu\bootstrap_gpu.py" 4 %*
)
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo FAILED. Inspect the launcher message and revision_results logs.
if not defined REVISION_NO_PAUSE pause
exit /b %RESULT%

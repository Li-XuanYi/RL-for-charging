@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYBAMM_DISABLE_TELEMETRY=true"
pushd "%~dp0"
if exist "%~dp0.venv-reproduce\Scripts\python.exe" (
  "%~dp0.venv-reproduce\Scripts\python.exe" "%~dp0revision_experiments\bootstrap.py" 1 %*
) else (
  python "%~dp0revision_experiments\bootstrap.py" 1 %*
)
set "REVISION_EXIT=%ERRORLEVEL%"
echo.
if "%REVISION_EXIT%"=="0" echo Computation completed. Inspect revision_results\experiment_1 for scientific assessment.
if not "%REVISION_EXIT%"=="0" echo Failed or interrupted. Inspect revision_results\experiment_1 logs. Experiment 2 requires a completed handoff.
popd
if "%~1"=="" pause
exit /b %REVISION_EXIT%

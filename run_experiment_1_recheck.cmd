@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYBAMM_DISABLE_TELEMETRY=true"
pushd "%~dp0" || exit /b 1
echo Reusing recorded E1 optimizer parameters. Recomputing finer-mesh predictions and original precision gates.
if exist "%~dp0.venv-reproduce\Scripts\python.exe" (
  "%~dp0.venv-reproduce\Scripts\python.exe" "%~dp0revision_experiments\bootstrap.py" 1 --config "%~dp0revision_experiments\configs\experiment_1_recheck.json" %*
) else if exist "%~dp0.venv-revision\Scripts\python.exe" (
  "%~dp0.venv-revision\Scripts\python.exe" "%~dp0revision_experiments\bootstrap.py" 1 --config "%~dp0revision_experiments\configs\experiment_1_recheck.json" %*
) else (
  python "%~dp0revision_experiments\bootstrap.py" 1 --config "%~dp0revision_experiments\configs\experiment_1_recheck.json" %*
)
set "REVISION_EXIT=%ERRORLEVEL%"
echo.
if "%REVISION_EXIT%"=="0" echo E1 recheck passed. You may now run run_experiment_2.cmd.
if not "%REVISION_EXIT%"=="0" echo Recheck did not pass. Read the new experiment_1 run.log and summary.json. Do not start E2 yet.
popd
if "%~1"=="" pause
exit /b %REVISION_EXIT%

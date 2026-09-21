@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYBAMM_DISABLE_TELEMETRY=true"
pushd "%~dp0" || exit /b 1
echo EXPLORATORY E2: 2500 steps per training, 2 seeds, 3 decision periods, 2 training modes.
echo Total training budget: 30000 pack decisions. Validation and tests are additional.
echo Results: revision_results\experiment_2_exploratory_2500_2seeds
if exist "%~dp0.venv-reproduce\Scripts\python.exe" (
  "%~dp0.venv-reproduce\Scripts\python.exe" "%~dp0revision_experiments\bootstrap.py" 2 --config "%~dp0revision_experiments\configs\experiment_2_exploratory_2500_2seeds.json" %*
) else if exist "%~dp0.venv-revision\Scripts\python.exe" (
  "%~dp0.venv-revision\Scripts\python.exe" "%~dp0revision_experiments\bootstrap.py" 2 --config "%~dp0revision_experiments\configs\experiment_2_exploratory_2500_2seeds.json" %*
) else (
  python "%~dp0revision_experiments\bootstrap.py" 2 --config "%~dp0revision_experiments\configs\experiment_2_exploratory_2500_2seeds.json" %*
)
set "REVISION_EXIT=%ERRORLEVEL%"
echo.
if "%REVISION_EXIT%"=="0" echo Exploratory computation completed. This is not a formal experiment pass.
if not "%REVISION_EXIT%"=="0" echo Failed or interrupted. Read logs under revision_results\experiment_2_exploratory_2500_2seeds.
popd
if "%~1"=="" pause
exit /b %REVISION_EXIT%

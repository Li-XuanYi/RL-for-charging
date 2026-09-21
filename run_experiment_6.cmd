@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYBAMM_DISABLE_TELEMETRY=true"
pushd "%~dp0" || exit /b 1
if not exist "%~dp0revision_experiments\control.py" goto missing_base
if defined REPRO_PYTHON (
  "%REPRO_PYTHON%" "%~dp0revision_extensions\bootstrap.py" 6 %*
) else if exist "%~dp0.venv-revision\Scripts\python.exe" (
  "%~dp0.venv-revision\Scripts\python.exe" "%~dp0revision_extensions\bootstrap.py" 6 %*
) else if exist "%~dp0.venv-reproduce\Scripts\python.exe" (
  "%~dp0.venv-reproduce\Scripts\python.exe" "%~dp0revision_extensions\bootstrap.py" 6 %*
) else (
  where py >nul 2>nul
  if errorlevel 1 (
    python "%~dp0revision_extensions\bootstrap.py" 6 %*
  ) else (
    py -3 "%~dp0revision_extensions\bootstrap.py" 6 %*
  )
)
set "REVISION_EXIT=%ERRORLEVEL%"
goto finish
:missing_base
echo Missing revision_experiments. Keep this CMD beside the E1/E2 code package.
set "REVISION_EXIT=1"
:finish
echo.
if "%REVISION_EXIT%"=="0" echo Computation finished. Read revision_results\experiment_6 scientific_assessment.json.
if not "%REVISION_EXIT%"=="0" echo Failed or interrupted. Inspect revision_results\experiment_6 logs and required upstream runs.
popd
if "%~1"=="" pause
exit /b %REVISION_EXIT%

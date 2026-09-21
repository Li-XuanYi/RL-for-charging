@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYBAMM_DISABLE_TELEMETRY=true"
pushd "%~dp0"
python "%~dp0mainline_reproduction\bootstrap_mainline.py" %*
set "MAINLINE_EXIT=%ERRORLEVEL%"
echo.
if "%MAINLINE_EXIT%"=="0" echo Completed. Results: "%~dp0mainline_results". See LATEST.txt.
if not "%MAINLINE_EXIT%"=="0" echo Failed or interrupted. See mainline_results logs.
popd
if "%~1"=="" pause
exit /b %MAINLINE_EXIT%

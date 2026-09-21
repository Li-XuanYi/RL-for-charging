@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
pushd "%~dp0"
where python >nul 2>nul
if errorlevel 1 goto use_py
python "%~dp0reproduction\bootstrap.py" %*
goto finished
:use_py
py -3 "%~dp0reproduction\bootstrap.py" %*
:finished
set "REPRO_EXIT=%ERRORLEVEL%"
echo.
if not "%REPRO_EXIT%"=="0" echo Reproduction failed. Review the messages above and reproduction_results logs.
if "%REPRO_EXIT%"=="0" echo Results are in "%~dp0reproduction_results". Latest run: reproduction_results\LATEST.txt
popd
if "%~1"=="" pause
exit /b %REPRO_EXIT%

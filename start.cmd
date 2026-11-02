@echo off
rem ============================================================================
rem  DocForensics starter (Windows)
rem    start.cmd            set up on first run, then launch the web app
rem    start.cmd setup      (re)install dependencies and prepare samples/model only
rem    start.cmd test       run the regression tests (pytest)
rem    start.cmd eval       run the full synthetic benchmark with pass/fail gates
rem    start.cmd check      check the semantic embedding model
rem    start.cmd cli PATH   run one investigation from the command line
rem                         (PATH = a file, a folder or a .zip; output goes to output\)
rem    start.cmd help       show this help
rem ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title DocForensics

set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"
if /i "%MODE%"=="help" goto :help
if /i "%MODE%"=="-h" goto :help
if /i "%MODE%"=="/?" goto :help

rem ---- 1. find Python (3.10 preferred; the project is developed and tested on it) -------------
set "PY="
where py >nul 2>nul && py -3.10 -c "import sys" >nul 2>nul && set "PY=py -3.10"
if not defined PY where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [ERROR] Python was not found. Install Python 3.10 from https://www.python.org/downloads/release/python-31011/
  echo         and tick "Add python.exe to PATH" ^(the py launcher is enough^).
  set "RC=1"
  goto :finish
)
set "PYVER="
for /f %%v in ('%PY% -c "import sys;print(sys.version_info.major, sys.version_info.minor, sep=chr(46))"') do set "PYVER=%%v"
echo [setup] Using Python %PYVER%
if not "%PYVER%"=="3.10" echo [NOTE] Python 3.10 is recommended ^(install it next to your current one; this script picks it automatically^).

rem ---- 2. virtual environment (rebuilt if it was made with a different Python version) --------
set "VENVVER="
if exist ".venv\Scripts\python.exe" for /f %%v in ('.venv\Scripts\python.exe -c "import sys;print(sys.version_info.major, sys.version_info.minor, sep=chr(46))"') do set "VENVVER=%%v"
if defined VENVVER if not "%VENVVER%"=="%PYVER%" call :rebuild_venv
if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment .venv with Python %PYVER% ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    set "RC=1"
    goto :finish
  )
)
set "VPY=%CD%\.venv\Scripts\python.exe"

rem ---- 3. dependencies ---------------------------------------------------------
rem The marker file is a COPY of requirements.txt from the last successful install, so any change to the
rem requirements (new or updated package) triggers a reinstall automatically.
if /i "%MODE%"=="setup" if exist ".venv\.deps_ok" del ".venv\.deps_ok"
set "DEPS_STALE="
if not exist ".venv\.deps_ok" set "DEPS_STALE=1"
fc /b requirements.txt ".venv\.deps_ok" >nul 2>nul || set "DEPS_STALE=1"
if defined DEPS_STALE (
  echo [setup] Installing dependencies. The first run can take a few minutes ...
  "%VPY%" -m pip install --upgrade pip -q
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection and try again.
    set "RC=1"
    goto :finish
  )
  copy /y requirements.txt ".venv\.deps_ok" >nul
)

rem ---- 3b. optional packages (a failure here only warns; the app still works on Tesseract) ----------
if exist requirements-optional.txt (
  if /i "%MODE%"=="setup" if exist ".venv\.opt_ok" del ".venv\.opt_ok"
  set "OPT_STALE="
  if not exist ".venv\.opt_ok" set "OPT_STALE=1"
  fc /b requirements-optional.txt ".venv\.opt_ok" >nul 2>nul || set "OPT_STALE=1"
  if defined OPT_STALE (
    echo [setup] Installing optional OCR engine ^(RapidOCR^) ...
    "%VPY%" -m pip install -r requirements-optional.txt
    if errorlevel 1 (
      echo [WARNING] Optional RapidOCR could not be installed. Busy-background ID cards will read worse; everything else works.
    ) else (
      copy /y requirements-optional.txt ".venv\.opt_ok" >nul
    )
  )
)

rem ---- 4. Tesseract OCR (separate Windows program, not a pip package) ---------
set "TESS_OK="
if defined TESSERACT_CMD if exist "%TESSERACT_CMD%" set "TESS_OK=1"
where tesseract >nul 2>nul && set "TESS_OK=1"
if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" set "TESS_OK=1"
if exist "%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe" set "TESS_OK=1"
if exist "%LocalAppData%\Programs\Tesseract-OCR\tesseract.exe" set "TESS_OK=1"
if not defined TESS_OK (
  echo.
  echo [WARNING] Tesseract OCR was not found, so documents cannot be read yet.
  echo           Install the Windows build from https://github.com/UB-Mannheim/tesseract/wiki
  echo           or set TESSERACT_CMD to the full path of tesseract.exe, then run this again.
  echo.
)

rem ---- 5. fictional sample documents ------------------------------------------
if not exist "samples\case_clean" (
  echo [setup] Generating fictional sample documents ...
  "%VPY%" synth\generate.py
  "%VPY%" synth\generate.py --hard
)

rem ---- 6. semantic model (optional, about 70 MB, downloaded once) -------------
if not exist "models\models--Qdrant--bge-small-en-v1.5-onnx-Q" (
  echo [setup] Downloading the semantic model once. Needs internet; the app still works without it.
  "%VPY%" -m docforensics.embed_classifier --check
)
if exist "models\models--Qdrant--bge-small-en-v1.5-onnx-Q" set "HF_HUB_OFFLINE=1"

rem ---- 7. what to do -----------------------------------------------------------
if /i "%MODE%"=="setup" (
  echo.
  echo Setup complete. Start the app with:  start.cmd
  set "RC=0"
  goto :finish
)
if /i "%MODE%"=="test" goto :do_test
if /i "%MODE%"=="eval" goto :do_eval
if /i "%MODE%"=="check" goto :do_check
if /i "%MODE%"=="cli" goto :do_cli
goto :do_run

:do_run
echo.
echo Starting DocForensics. Your browser opens at http://localhost:8501  -  press Ctrl+C here to stop.
"%VPY%" -m streamlit run app.py
set "RC=%ERRORLEVEL%"
goto :finish

:do_test
"%VPY%" -m pytest -q
set "RC=%ERRORLEVEL%"
goto :finish

:do_eval
"%VPY%" tests\eval_synth.py
set "RC=%ERRORLEVEL%"
goto :finish

:do_check
"%VPY%" -m docforensics.embed_classifier --check
set "RC=%ERRORLEVEL%"
goto :finish

:do_cli
if "%~2"=="" (
  echo Usage: start.cmd cli PATH   - PATH is a file, a folder or a .zip
  set "RC=1"
  goto :finish
)
"%VPY%" -m docforensics.pipeline %2 %3 %4 %5
set "RC=%ERRORLEVEL%"
goto :finish

:help
setlocal EnableDelayedExpansion
for /f "usebackq delims=" %%a in (`findstr /b /c:"rem  " "%~f0"`) do (
  set "L=%%a"
  echo !L:~4!
)
endlocal
set "RC=0"
goto :finish

:rebuild_venv
echo [setup] The existing .venv uses Python %VENVVER% but %PYVER% is selected - rebuilding it ...
rmdir /s /q ".venv"
exit /b 0

:finish
echo.
if not "%RC%"=="0" echo Finished with exit code %RC%.
pause
exit /b %RC%

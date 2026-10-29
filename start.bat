@echo off
rem Starts the Swalker demo: backend (port 8000) and frontend (port 5173), each in its own
rem window, then opens the site once the backend answers. Close those two windows to stop it.
rem Double-click it, or run "start.bat" from the repo folder in cmd.

setlocal
cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
    echo The backend's Python environment is missing: backend\.venv
    echo Set it up first, see backend\README.md, "Setup".
    goto :fail
)
if not exist "backend\models\stage2_iter70000.pth" (
    echo The model checkpoint is missing: backend\models\stage2_iter70000.pth
    echo Download it from the Kaggle dataset bbiiggnn/gait-checkpoints, see backend\README.md.
    goto :fail
)
where npm >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Install Node.js, then run this again.
    goto :fail
)
if not exist "frontend\node_modules" (
    echo Installing the frontend's packages, first run only...
    pushd frontend
    call npm install
    if errorlevel 1 (
        popd
        goto :fail
    )
    popd
)

start "Swalker backend" /d "%~dp0backend" cmd /k ".venv\Scripts\python -m uvicorn app.main:create_app --factory --port 8000"
start "Swalker frontend" /d "%~dp0frontend" cmd /k "npm run dev"

rem Loading YOLO and GLTNet takes a while; wait up to about 2 minutes for the backend.
echo Waiting for the backend to load the models...
set /a tries=0
:wait
curl -sf http://localhost:8000/api/health >nul 2>nul
if not errorlevel 1 goto :ready
set /a tries+=1
if %tries% geq 60 (
    echo The backend did not answer after 2 minutes. Look at the "Swalker backend" window for the error.
    goto :fail
)
timeout /t 2 /nobreak >nul
goto :wait

:ready
echo Backend is up. Opening http://localhost:5173
start "" http://localhost:5173
echo Model status: http://localhost:8000/api/health ("modelLoaded" should be true)
endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1

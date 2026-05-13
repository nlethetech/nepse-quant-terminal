@echo off
setlocal

cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3.12"
) else (
    python --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_CMD=python"
    ) else (
        echo Python was not found.
        echo Install Python 3.12 from https://www.python.org/downloads/ and launch again.
        echo Make sure "Add python.exe to PATH" is selected during install.
        echo.
        pause
        exit /b 1
    )
)

%PYTHON_CMD% scripts\launch\bootstrap.py
set "STATUS=%ERRORLEVEL%"

echo.
if not "%STATUS%"=="0" (
    echo NEPSE Quant Terminal exited with status %STATUS%.
    echo Read the message above, fix the issue, then launch again.
) else (
    echo NEPSE Quant Terminal closed.
)
echo.
pause
exit /b %STATUS%

@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
set "BUILD=%ROOT%\build"
set "BIN_DIR=%BUILD%\bin\Debug"

if not exist "%BUILD%\CMakeCache.txt" (
    echo [run] Initializing CMake...
    cmake -S "%ROOT%" -B "%BUILD%"
    if errorlevel 1 exit /b 1
)

if "%~1"=="" (
    set "TARGET=scratch"
) else (
    set "TARGET=%~n1"
    if "%~x1"=="" set "TARGET=%~1"
)

echo [run] Building: %TARGET%
cmake --build "%BUILD%" --target %TARGET% --config Debug
if errorlevel 1 (
    echo [run] BUILD FAILED
    exit /b 1
)

set "EXE=%BIN_DIR%\%TARGET%.exe"
if not exist "%EXE%" (
    echo [run] %EXE% not found, searching...
    for /r "%BUILD%" %%F in (%TARGET%.exe) do set "EXE=%%F"
)

if not exist "%EXE%" (
    echo [run] Cannot find %TARGET%.exe
    exit /b 1
)

echo.
echo ============================================================
"%EXE%"
echo ============================================================
echo [run] Exit code: %ERRORLEVEL%
endlocal
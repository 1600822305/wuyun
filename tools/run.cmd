@echo off
REM ============================================================
REM  WuYun 快速实验 - 一键编译运行 scratch.cpp
REM
REM  用法:
REM      tools\run.cmd              编译并运行 scratch.cpp
REM      tools\run.cmd --build-only 只编译不运行
REM      tools\run.cmd --run-only   只运行(跳过编译)
REM ============================================================

setlocal
set "ROOT=%~dp0.."
set "BUILD=%ROOT%\build"
set "TARGET=scratch"

REM 解析参数
set "DO_BUILD=1"
set "DO_RUN=1"
if "%1"=="--build-only" (set "DO_RUN=0")
if "%1"=="--run-only" (set "DO_BUILD=0")

REM 确保 build 目录存在
if not exist "%BUILD%" (
    echo [scratch] 首次运行, 初始化 CMake...
    cmake -S "%ROOT%" -B "%BUILD%"
    if errorlevel 1 (
        echo [scratch] CMake 配置失败!
        exit /b 1
    )
)

REM 编译
if "%DO_BUILD%"=="1" (
    echo [scratch] 编译中...
    cmake --build "%BUILD%" --target %TARGET% --config Debug
    if errorlevel 1 (
        echo.
        echo [scratch] !! 编译失败 !!
        exit /b 1
    )
    echo [scratch] 编译成功
    echo.
)

REM 运行
if "%DO_RUN%"=="1" (
    echo ============================================================
    set "EXE=%BUILD%\bin\%TARGET%.exe"
    if exist "%EXE%" (
        "%EXE%"
    ) else (
        REM MSVC 多配置生成器可能放在 Debug 子目录
        set "EXE=%BUILD%\bin\Debug\%TARGET%.exe"
        if exist "%BUILD%\bin\Debug\%TARGET%.exe" (
            "%BUILD%\bin\Debug\%TARGET%.exe"
        ) else (
            echo [scratch] 找不到可执行文件, 尝试的路径:
            echo   %BUILD%\bin\%TARGET%.exe
            echo   %BUILD%\bin\Debug\%TARGET%.exe
            exit /b 1
        )
    )
    echo.
    echo ============================================================
    echo [scratch] 退出码: %ERRORLEVEL%
)

endlocal
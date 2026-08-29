@echo off
REM BharatTerminal Smart Collector Launcher
REM Runs at Windows login. No terminal server, Tauri, or Redis required.

set "COMPUTE_DIR=%~dp0.."
cd /d "%COMPUTE_DIR%"

if not defined BHARATTERMINAL_APP_DATA set "BHARATTERMINAL_APP_DATA=%COMPUTE_DIR%\..\.."

REM Prefer uv's own standard self-installer location over whatever "uv.exe"
REM resolves to first on PATH. On this machine PATH puts a session/AI-agent
REM tooling directory (...\hermes\bin\uv.exe) ahead of the durable,
REM user-owned install at %USERPROFILE%\.local\bin\uv.exe -- a scheduled
REM task must not depend on that tooling still being present at some
REM future login. Falls back to PATH resolution for machines without this
REM standard install path.
set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if exist "%UV_EXE%" goto :uv_resolved

set "UV_EXE=uv.exe"
where uv.exe >nul 2>nul
if errorlevel 1 (
    echo Canonical uv runtime launcher not found
    exit /b 1
)

:uv_resolved

REM Launch the Job Object supervisor, not the collector directly: it owns a
REM Windows Job Object (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) so force-killing
REM this top-level process tree cannot leave an orphaned python.exe behind.
"%UV_EXE%" run --project "%COMPUTE_DIR%" --frozen python "auto_collector\job_supervisor.py"
exit /b %ERRORLEVEL%

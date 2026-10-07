@echo off
powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0run_scenarioview_daily.ps1" %*
exit /b %errorlevel%

@echo off
rem Arrete le serveur Streamlit lance en silencieux (port 8501 par defaut).
setlocal enabledelayedexpansion
set FOUND=0
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo Arret du processus %%p...
    taskkill /F /PID %%p >nul 2>&1
    set FOUND=1
)
if "!FOUND!"=="0" (
    echo Aucun serveur Clip Creator trouve sur le port 8501.
) else (
    echo Clip Creator arrete.
)
pause

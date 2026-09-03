@echo off
setlocal
cd /d "%~dp0.."

rem Utilise l'environnement virtuel present : venv\ (celui du projet) en priorite, sinon .venv\
set "PYEXE="
if exist "venv\Scripts\python.exe" set "PYEXE=venv\Scripts\python.exe"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"

if not defined PYEXE (
    echo [Clip Creator] Environnement virtuel introuvable ^(venv\ ou .venv\^).
    echo Depuis le dossier du projet, dans une invite de commandes :
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYEXE%" main.py

echo.
echo [Clip Creator] Le serveur s'est arrete.
pause

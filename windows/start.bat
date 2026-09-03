@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [Clip Creator] Environnement virtuel introuvable dans .venv
    echo Lancez d'abord dans une invite de commandes, depuis le dossier du projet :
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python main.py

echo.
echo [Clip Creator] Le serveur s'est arrete.
pause

@echo off
title Orchestr'IA
echo ========================================
echo        Lancement d'Orchestr'IA
echo ========================================
echo.

:: Chercher Python
where python >nul 2>nul && set PYTHON=python && goto :found
where python3 >nul 2>nul && set PYTHON=python3 && goto :found
echo ERREUR : Python n'est pas installe ou n'est pas dans le PATH.
echo Installez Python depuis https://www.python.org/downloads/
pause
exit /b 1

:found

:: Se placer dans le dossier du script
cd /d "%~dp0"

:: Verifier que les dependances sont installees
echo Verification des dependances...
%PYTHON% -m pip install -r requirements.txt --quiet
echo Dependances OK.
echo.

:: Lancer l'application
echo Demarrage du serveur Streamlit...
echo L'application va s'ouvrir dans votre navigateur.
echo Pour arreter : fermez cette fenetre ou appuyez sur Ctrl+C.
echo.
%PYTHON% -m streamlit run src/app.py
pause

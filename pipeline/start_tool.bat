@echo off
:: Wechsle in das Verzeichnis, in dem diese .bat-Datei liegt
cd /D %~dp0

ECHO ========================================================
ECHO          Review Analyse Tool - Starter
ECHO ========================================================
ECHO.
ECHO Arbeitsverzeichnis: %CD%
ECHO.

ECHO Pruefe, ob Python verfuegbar ist...
python --version > NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO FEHLER: Python scheint nicht installiert zu sein.
    PAUSE
    EXIT /B
)
ECHO Python gefunden.
ECHO.

ECHO Erstelle virtuelle Umgebung (venv), falls nicht vorhanden...
IF NOT EXIST venv (
    ECHO Richte 'venv' ein...
    python -m venv venv
) ELSE (
    ECHO 'venv' existiert bereits, nutze bestehende.
)
ECHO.

ECHO Aktiviere virtuelle Umgebung...
CALL venv\Scripts\activate.bat
ECHO.

ECHO ========================================================
ECHO SCHRITT 1: Aktualisiere PIP (den Paket-Installer)
ECHO ========================================================
python -m pip install --upgrade pip
ECHO.

ECHO ========================================================
ECHO SCHRITT 2: Installiere Pakete (ohne Cache)...
ECHO ========================================================
pip install --no-cache-dir -r requirements.txt
ECHO.

ECHO ========================================================
ECHO SCHRITT 3: Lade deutsches Spacy-Modell...
ECHO ========================================================
python -m spacy download de_core_news_sm
ECHO.

ECHO ========================================================
ECHO Installation abgeschlossen. Starte das Tool...
ECHO Das Tool oeffnet sich gleich in deinem Webbrowser.
ECHO ========================================================
ECHO.

streamlit run app.py

ECHO.
ECHO Das Tool wurde beendet. Du kannst dieses Fenster schliessen.
PAUSE
#!/bin/bash

# NEU: Wechsle in das Verzeichnis, in dem dieses Skript liegt
cd "$(dirname "$0")"

echo "========================================================"
echo "         Review Analyse Tool - Starter (macOS)"
echo "========================================================"
echo "Arbeitsverzeichnis: $(pwd)"
echo

echo "Pruefe, ob Python 3 verfuegbar ist..."
if ! command -v python3 &> /dev/null
then
    echo "FEHLER: 'python3' wurde nicht gefunden."
    exit 1
fi
echo "Python 3 gefunden."
echo

echo "Erstelle virtuelle Umgebung (venv), falls nicht vorhanden..."
if [ ! -d "venv" ]
then
    echo "Richte 'venv' ein..."
    python3 -m venv venv
else
    echo "'venv' existiert bereits, nutze bestehende."
fi
echo

echo "Aktiviere virtuelle Umgebung..."
source venv/bin/activate
echo

echo "========================================================"
echo "SCHRITT 1: Aktualisiere PIP (den Paket-Installer)"
echo "========================================================"
pip3 install --upgrade pip
echo

echo "========================================================"
echo "SCHRITT 2: Installiere Pakete (ohne Cache)..."
echo "========================================================"
pip3 install --no-cache-dir -r requirements.txt
echo

echo "========================================================"
echo "SCHRITT 3: Lade deutsches Spacy-Modell..."
echo "========================================================"
python3 -m spacy download de_core_news_sm
echo

echo "========================================================"
echo "Installation abgeschlossen. Starte das Tool..."
echo "========================================================"
echo

streamlit run app.py

# Aufräumen
deactivate
echo "Tool beendet."
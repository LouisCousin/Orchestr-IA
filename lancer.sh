#!/bin/bash
# Orchestr'IA - Script de lancement (Linux/Mac)

echo "========================================"
echo "       Lancement d'Orchestr'IA"
echo "========================================"
echo

cd "$(dirname "$0")"

# Chercher Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERREUR : Python n'est pas installé."
    echo "Installez Python 3.9+ depuis https://www.python.org/downloads/"
    exit 1
fi

# Vérifier les dépendances
echo "Vérification des dépendances..."
$PYTHON -m pip install -r requirements.txt --quiet
echo "Dépendances OK."
echo

echo "Démarrage du serveur Streamlit..."
echo "L'application va s'ouvrir dans votre navigateur."
echo "Pour arrêter : Ctrl+C"
echo
$PYTHON -m streamlit run src/app.py

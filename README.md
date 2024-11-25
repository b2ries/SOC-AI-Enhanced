# SOC-AI-Enhanced
## Description
SOC-AI-Enhanced est un projet d'intelligence artificielle conçu pour améliorer la sécurité des opérations dans un Security Operations Center (SOC). Ce projet utilise des techniques de deep learning et des modèles de traitement du langage naturel pour analyser les logs de sécurité et détecter les menaces.

## Structure du Projet
- `data/`: Contient les jeux de données et les logs de sécurité.
- `notebooks/`: Jupyter Notebooks pour l'exploration des données.
- `src/`: Code source du projet.
  - `data_preprocessing.py`: Prétraitement des données.
  - `model.py`: Modèle de deep learning.
  - `utils.py`: Fonctions utilitaires.
  - `main.py`: Fichier principal pour exécuter le projet.
- `requirements.txt`: Dépendances du projet.

## Installation
Pour installer les dépendances, exécutez :
pip install -r requirements.txt

## Utilisation
1. Prétraiter les logs de sécurité en exécutant `data_preprocessing.py`.
2. Entraîner le modèle en utilisant `model.py` ou exécutez `main.py` pour automatiser le processus.
3. Utiliser les fonctions utilitaires dans `utils.py` pour évaluer le modèle.

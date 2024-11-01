# Velocity Predictor

Ce projet utilise un modèle de régression linéaire pour prédire la vélocité d'un sprint en fonction des jours-homme disponibles. La vélocité est calculée sur la base des données simulées pour des équipes de développement agiles, et les prédictions peuvent être faites en fonction de la taille de l'équipe et du nombre de jours ouvrés dans le sprint.

## Structure du Projet

Le projet est composé de trois scripts Python :
1. **`create_fake_data.py`** : Génère des données simulées de jours-homme disponibles et de vélocité, puis les enregistre dans un fichier Parquet. Ces données servent à entraîner le modèle de prédiction.
2. **`fit_valid_dump.py`** : Entraîne un modèle de régression linéaire sur les données générées, normalise les données pour faciliter la prédiction, et enregistre le modèle entraîné et le scaler pour une utilisation future.
3. **`predict_velocity.py`** : Utilise le modèle et le scaler enregistrés pour prédire la vélocité d'un sprint, en fonction des informations fournies par l'utilisateur sur la taille de l'équipe et les jours ouvrés disponibles.

---

## Fichiers Python

### 1. `create_fake_data.py`

Ce script génère des données factices pour entraîner le modèle de prédiction de vélocité. Il permet de configurer le nombre de sprints, la taille de l'équipe, et le nombre de jours ouvrés maximum par personne. La vélocité est simulée en tant que fonction des jours-homme disponibles, avec une légère variabilité aléatoire pour simuler des fluctuations naturelles dans la productivité.

**Fonctions principales :**
- **`generate_data()`** : Génère un DataFrame avec les jours-homme disponibles et une vélocité simulée.
- **`plot_data()`** : Affiche un graphique de dispersion pour vérifier la relation entre les jours-homme disponibles et la vélocité.
- **`save_to_parquet()`** : Sauvegarde le DataFrame généré dans un fichier Parquet.

**Utilisation :**
Exécute le script pour générer et sauvegarder un fichier `data/fake_data.parquet` contenant les données de simulation.
```bash
python create_fake_data.py
```

### 2. `fit_valid_dump.py`

Ce script entraîne un modèle de régression linéaire sur les données générées et normalise les valeurs d'entrée pour améliorer les prédictions. Il utilise les jours-homme disponibles pour prédire la vélocité d'un sprint. Le modèle entraîné ainsi que le scaler sont enregistrés pour être utilisés dans le script de prédiction.

**Fonctions principales :**
- **`load_data()`** : Charge les données depuis le fichier Parquet généré.
- **`preprocess_data()`** : Divise les données en ensembles d'entraînement et de test, et les normalise.
- **`train_linear_regression()`** : Entraîne le modèle de régression linéaire.
- **`evaluate_model()`** : Évalue la précision du modèle.
- **`save_model()`** : Enregistre le modèle et le scaler dans le dossier `model/`.

**Utilisation :**
Exécute le script pour entraîner et sauvegarder le modèle.
```bash
python fit_valid_dump.py
```

### 3. `predict_velocity.py`

Ce script charge le modèle et le scaler sauvegardés, puis utilise des données entrées par l'utilisateur pour prédire la vélocité d'un sprint. L'utilisateur est invité à entrer le nombre de personnes dans l'équipe et le nombre de jours ouvrés du sprint. Le script calcule ensuite les jours-homme disponibles, normalise cette valeur et utilise le modèle pour faire une prédiction de vélocité.

**Fonctions principales :**
- **`load_model_and_scaler()`** : Charge le modèle de régression linéaire et le scaler depuis le dossier `model/`.
- **`get_user_input()`** : Demande à l'utilisateur d'entrer le nombre de personnes dans l'équipe et les jours ouvrés.
- **`predict_velocity()`** : Calcule la vélocité prévue en fonction des jours-homme disponibles et affiche la prédiction.

**Utilisation :**
Exécute le script pour obtenir une prédiction de vélocité basée sur les données entrées.
```bash
python predict_velocity.py
```

---

## Prérequis

- Python 3.x
- Bibliothèques Python : `pandas`, `numpy`, `scikit-learn`, `duckdb`, `seaborn`, `matplotlib`, `joblib`

Installez les dépendances avec la commande suivante :
```bash
pip install -r requirements.txt
```

## Exemple d'Utilisation

1. **Générer les données** : Utilisez `create_fake_data.py` pour créer un fichier de données simulées dans `data/fake_data.parquet`.
2. **Entraîner le modèle** : Exécutez `fit_valid_dump.py` pour entraîner le modèle et sauvegarder le modèle et le scaler.
3. **Faire une prédiction** : Exécutez `predict_velocity.py` et suivez les instructions pour prédire la vélocité en fonction des jours-homme disponibles dans un sprint.
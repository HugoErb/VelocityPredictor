import os

import duckdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Parquet en utilisant DuckDB.

    :param file_path: Chemin du fichier Parquet contenant les données.
    :type file_path: str
    :return: DataFrame contenant les données chargées.
    :rtype: pd.DataFrame
    """
    conn = duckdb.connect(database=':memory:', read_only=False)
    conn.execute(f"CREATE TABLE my_table AS SELECT * FROM parquet_scan('{file_path}')")
    df = conn.execute("SELECT * FROM my_table").fetch_df()
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Prépare les données pour l'entraînement en normalisant et divisant en ensembles train/test.

    :param df: DataFrame contenant les données d'origine.
    :type df: pd.DataFrame
    :return: Un tuple contenant les données d'entraînement et de test normalisées, ainsi que le scaler.
    :rtype: tuple
    """
    X = df[['jours_homme_dispo']]
    y = df['velocite_reelle']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Entraîne un modèle de régression linéaire.

    :param X_train: Données d'entraînement normalisées.
    :type X_train: np.ndarray
    :param y_train: Valeurs cibles pour l'entraînement.
    :type y_train: np.ndarray
    :return: Modèle de régression linéaire entraîné.
    :rtype: LinearRegression
    """
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    return linear_model


def transform_polynomial(X: np.ndarray, degree: int) -> tuple:
    """
    Transforme les données d'entrée en utilisant des caractéristiques polynomiales.

    :param X: Données d'entrée normalisées.
    :type X: np.ndarray
    :param degree: Degré des caractéristiques polynomiales.
    :type degree: int
    :return: Les données transformées et l'objet PolynomialFeatures.
    :rtype: tuple
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


def train_polynomial_regression(X_train_poly: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Entraîne un modèle de régression linéaire sur des caractéristiques polynomiales.

    :param X_train_poly: Données d'entraînement transformées en caractéristiques polynomiales.
    :type X_train_poly: np.ndarray
    :param y_train: Valeurs cibles pour l'entraînement.
    :type y_train: np.ndarray
    :return: Modèle de régression polynomiale entraîné.
    :rtype: LinearRegression
    """
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    return poly_model


def evaluate_model(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray,
                   poly: PolynomialFeatures = None) -> dict:
    """
    Évalue les performances d'un modèle de régression et renvoie les erreurs.

    :param model: Modèle de régression à évaluer.
    :type model: LinearRegression
    :param X_test: Données de test.
    :type X_test: np.ndarray
    :param y_test: Valeurs cibles pour les données de test.
    :type y_test: np.ndarray
    :param poly: Transformateur PolynomialFeatures pour transformer les données si applicable, par défaut None.
    :type poly: PolynomialFeatures, optional
    :return: Dictionnaire contenant les erreurs absolue et quadratique moyenne, ainsi que leurs pourcentages.
    :rtype: dict
    """
    if poly:
        X_test = poly.transform(X_test)

    predictions = model.predict(X_test)
    mean_velocity = y_test.mean()

    mae = mean_absolute_error(y_test, predictions)
    mae_percent = (mae / mean_velocity) * 100 if mean_velocity != 0 else 0
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    rmse_percent = (rmse / mean_velocity) * 100 if mean_velocity != 0 else 0

    return {
        "mae": mae,
        "mae_percent": mae_percent,
        "rmse": rmse,
        "rmse_percent": rmse_percent
    }


def save_model(model: object, filename: str):
    """
    Sauvegarde un modèle ou un transformateur dans un fichier, dans le dossier 'model'.
    Crée le dossier s'il n'existe pas.

    :param model: Modèle ou transformateur à sauvegarder.
    :type model: object
    :param filename: Nom du fichier de sauvegarde.
    :type filename: str
    """
    # On vérifie si le dossier 'model' existe, sinon on le crée
    directory = "model"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # On enregistre le modèle dans le dossier 'model'
    file_path = os.path.join(directory, filename)
    joblib.dump(model, file_path)


def main(use_polynomial_regression: bool = False):
    """
    Fonction principale pour charger les données, entraîner les modèles, évaluer les performances et sauvegarder les modèles.

    :param use_polynomial_regression: Indicateur pour utiliser ou non la régression polynomiale.
    :type use_polynomial_regression: bool
    """
    # Chargement et préparation des données
    df = load_data('data/fake_data.parquet')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Entraînement et évaluation de la régression linéaire
    linear_model = train_linear_regression(X_train, y_train)
    linear_results = evaluate_model(linear_model, X_test, y_test)

    print(f"Vélocité moyenne = {round(df['velocite_reelle'].mean(), 2)}\n")
    print("Évaluation de la régression linéaire :")
    print(f"Erreur absolue moyenne = {round(linear_results['mae'], 2)}")
    print(f"Erreur absolue moyenne en % = {round(linear_results['mae_percent'], 2)} %")
    print(f"Erreur quadratique moyenne = {round(linear_results['rmse'], 2)}")
    print(f"Erreur quadratique moyenne en % = {round(linear_results['rmse_percent'], 2)} %\n")

    # Enregistrement du modèle linéaire et du scaler
    save_model(linear_model, 'linear_model.joblib')
    save_model(scaler, 'scaler.joblib')

    # Entraînement et évaluation de la régression polynomiale
    if use_polynomial_regression:
        X_train_poly, poly = transform_polynomial(X_train, degree=4)
        polynomial_model = train_polynomial_regression(X_train_poly, y_train)
        polynomial_results = evaluate_model(polynomial_model, X_test, y_test, poly=poly)

        print("Évaluation de la régression polynomiale :")
        print(f"Erreur absolue moyenne = {round(polynomial_results['mae'], 2)}")
        print(f"Erreur absolue moyenne en % = {round(polynomial_results['mae_percent'], 2)} %")
        print(f"Erreur quadratique moyenne = {round(polynomial_results['rmse'], 2)}")
        print(f"Erreur quadratique moyenne en % = {round(polynomial_results['rmse_percent'], 2)} %\n")

        # Enregistrement du modèle polynomial et du transformateur
        save_model(polynomial_model, 'polynomial_model.joblib')
        save_model(poly, 'polynomial_features.joblib')


if __name__ == "__main__":
    main(use_polynomial_regression=False)

import joblib
import pandas as pd


def load_model_and_scaler():
    """
    Charge le modèle de régression linéaire et le scaler depuis le dossier 'model'.

    :return: Le modèle de régression linéaire et le scaler.
    :rtype: tuple
    """
    model = joblib.load('model/linear_model.joblib')
    scaler = joblib.load('model/scaler.joblib')
    return model, scaler


def get_user_input():
    """
    Demande à l'utilisateur d'entrer le nombre de personnes dans l'équipe et le nombre de jours ouvrés.

    :return: Les jours-homme disponibles calculés.
    :rtype: float
    """
    try:
        personnes = int(input("Entrez le nombre de personnes dans l'équipe : "))
        jours_ouvres = int(input("Entrez le nombre de jours ouvrés pendant l'itération : "))
        jours_homme_dispo = personnes * jours_ouvres
        return jours_homme_dispo
    except ValueError:
        print("Veuillez entrer des nombres valides pour le nombre de personnes et les jours ouvrés.")
        return None


def predict_velocity(model, scaler, jours_homme_dispo):
    """
    Prédit la vélocité du sprint en fonction des jours-homme disponibles.

    :param model: Modèle de régression linéaire chargé.
    :type model: LinearRegression
    :param scaler: Scaler utilisé pour normaliser les données.
    :type scaler: MinMaxScaler
    :param jours_homme_dispo: Nombre de jours-homme disponibles.
    :type jours_homme_dispo: float
    :return: Prédiction de la vélocité.
    :rtype: float
    """
    # Crée un DataFrame avec le même nom de colonne que lors de l'entraînement
    jours_homme_dispo_df = pd.DataFrame({"jours_homme_dispo": [jours_homme_dispo]})

    # Mise à l'échelle de la nouvelle donnée
    jours_homme_dispo_scaled = scaler.transform(jours_homme_dispo_df)

    # Prédiction avec le modèle
    predicted_velocity = model.predict(jours_homme_dispo_scaled)
    return predicted_velocity[0]


def main():
    """
    Programme principal.
    """
    # Charger le modèle et le scaler
    model, scaler = load_model_and_scaler()

    # Obtenir les jours-homme disponibles de l'utilisateur
    jours_homme_dispo = get_user_input()
    if jours_homme_dispo is None:
        return  # Arrête le programme si les entrées ne sont pas valides

    # Prédire la vélocité
    predicted_velocity = predict_velocity(model, scaler, jours_homme_dispo)
    print(f"Prédiction de la vélocité du sprint : {predicted_velocity:.2f} points")


if __name__ == "__main__":
    main()

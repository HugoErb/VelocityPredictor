import joblib
import numpy as np

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
        print(f"Jours-homme disponibles pour l'itération : {jours_homme_dispo}")
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
    jours_homme_dispo_scaled = scaler.transform([[jours_homme_dispo]])
    predicted_velocity = model.predict(jours_homme_dispo_scaled)
    return predicted_velocity[0]

def main():
    model, scaler = load_model_and_scaler()

    jours_homme_dispo = get_user_input()
    if jours_homme_dispo is None:
        return

    predicted_velocity = predict_velocity(model, scaler, jours_homme_dispo)
    print(f"Prédiction de la vélocité du sprint : {predicted_velocity:.2f} points")

if __name__ == "__main__":
    main()

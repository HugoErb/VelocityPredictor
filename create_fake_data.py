import os
import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Paramètres globaux
NB_SPRINTS = 1000  # Nombre de sprints à générer
MAX_JOURS = 15  # Maximum de jours par personne par sprint
NB_PERSONNES = 4  # Nombre de personnes dans l'équipe
MULTIPLICATEUR_VELOCITE = 2  # Facteur multiplicatif pour simuler la vélocité
np.random.seed(0)  # pour la reproductibilité aléatoire


def generate_data(nb_sprints, max_jours, nb_personnes, multiplicateur_velocite):
    """
    Génère un DataFrame avec des données de jours-homme disponibles et une vélocité réelle.

    :param nb_sprints: Nombre de sprints à générer.
    :param max_jours: Maximum de jours par personne par sprint.
    :param nb_personnes: Nombre de personnes dans l'équipe.
    :param multiplicateur_velocite: Facteur multiplicatif pour simuler la vélocité.
    :return: DataFrame avec les colonnes 'jours_homme_dispo' et 'velocite_reelle'.
    """
    jours_homme_dispo = np.random.randint(nb_personnes, nb_personnes * max_jours + 1, nb_sprints)
    variation_aleatoire = np.random.uniform(0.85, 1.15, nb_sprints)
    velocite_reelle = jours_homme_dispo * multiplicateur_velocite * variation_aleatoire
    return pd.DataFrame({"jours_homme_dispo": jours_homme_dispo, "velocite_reelle": velocite_reelle})


def plot_data(df):
    """
    Affiche un graphique de dispersion pour vérifier la relation entre les jours-homme et la vélocité.

    :param df: DataFrame contenant les données générées.
    """
    sns.scatterplot(data=df, x="jours_homme_dispo", y="velocite_reelle")
    plt.title("Relation entre jours-homme disponibles et vélocité réelle")
    plt.xlabel("Jours-homme disponibles")
    plt.ylabel("Vélocité réelle")
    plt.show()


def save_to_parquet(df, file_path):
    """
    Sauvegarde le DataFrame dans un fichier Parquet.

    :param df: DataFrame à sauvegarder.
    :param file_path: Chemin complet du fichier Parquet.
    """
    # Vérification que le dossier existe
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Sauvegarde avec DuckDB
    con = duckdb.connect(database=':memory:', read_only=False)
    con.register('my_table', df)
    con.execute(f"COPY my_table TO '{file_path}' (FORMAT 'PARQUET')")
    con.close()


def main():
    """
    Fonction principale
    """
    # Génération des données
    df = generate_data(NB_SPRINTS, MAX_JOURS, NB_PERSONNES, MULTIPLICATEUR_VELOCITE)

    # Vérification visuelle
    print(df.head())
    plot_data(df)

    # Sauvegarde en fichier Parquet
    save_to_parquet(df, 'data/fake_data.parquet')
    print("Données générées et sauvegardées dans 'data/fake_data.parquet'.")


if __name__ == "__main__":
    main()

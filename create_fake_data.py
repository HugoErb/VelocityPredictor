import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Paramètres
NB_SPRINTS = 1000  # Nombre de sprints à générer
MAX_JOURS = 15  # Maximum de jours par personne par sprint
NB_PERSONNES = 4  # Nombre de personnes dans l'équipe
np.random.seed(0)  # pour la reproductibilité aléatoire

# Génération de données
jours_homme_dispo = np.random.randint(NB_PERSONNES, NB_PERSONNES*MAX_JOURS + 1, NB_SPRINTS)

# Logique pour créer une vélocité réelle en fonction du nombre de jours homme disponibles
# Vélocité = 2 fois le nombre de jours homme + variation aléatoire
variation_aleatoire = np.random.uniform(0.85, 1.15, NB_SPRINTS)
velocite_reelle = jours_homme_dispo * variation_aleatoire

# Création du DataFrame
df = pd.DataFrame({"jours_homme_dispo": jours_homme_dispo, "velocite_reelle": velocite_reelle})

# Vérification des données générées
# print(df.head())
# sns.scatterplot(df, x=jours_homme_dispo, y=velocite_reelle)
# plt.show()

# Création du fichier parquet
con = duckdb.connect(database=':memory:', read_only=False)
con.register('my_table', df)
con.execute("COPY my_table TO 'data/fake_data.parquet' (FORMAT 'PARQUET')")
con.close()

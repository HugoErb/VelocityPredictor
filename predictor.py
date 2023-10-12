import duckdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# Récupération des données historiques dans le fichier parquet
conn = duckdb.connect(database=':memory:', read_only=False)
conn.execute("CREATE TABLE my_table AS SELECT * FROM parquet_scan('data/fake_data.parquet')")
query = "SELECT * FROM my_table"
df = conn.execute(query).fetch_df()
# print(df)

# Décomposition train/test
X = df[['jours_homme_dispo']]
y = df['velocite_reelle']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Transformation des données pour la régression polynomiale
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Entraînement avec regréssion linéaire
linear_model = LinearRegression()
linear_model_poly = LinearRegression()
linear_model.fit(X_train, y_train)
linear_model_poly.fit(X_train_poly, y_train)

# Prédictions
test_linear_predictions = linear_model.predict(X_test)
test_poly_predictions = linear_model_poly.predict(X_test_poly)

# Calcul des erreurs
mean_velocity = df['velocite_reelle'].mean()

mean_linear_absolute_error_value = mean_absolute_error(y_test, test_linear_predictions)
mean_linear_absolute_error_value_percent = (mean_linear_absolute_error_value / mean_velocity) * 100 if mean_velocity != 0 else 0
mean_linear_squared_error_value = np.sqrt(mean_squared_error(y_test, test_linear_predictions))
mean_linear_squared_error_value_percent = (mean_linear_squared_error_value / mean_velocity) * 100 if mean_velocity != 0 else 0

mean_poly_absolute_error_value = mean_absolute_error(y_test, test_poly_predictions)
mean_poly_absolute_error_value_percent = (mean_poly_absolute_error_value / mean_velocity) * 100 if mean_velocity != 0 else 0
mean_poly_squared_error_value = np.sqrt(mean_squared_error(y_test, test_poly_predictions))
mean_poly_squared_error_value_percent = (mean_poly_squared_error_value / mean_velocity) * 100 if mean_velocity != 0 else 0

print(f'Vélocité moyenne = {round(mean_velocity,2)}')
print('')
print(f'Erreur linéaire absolue moyenne flat = {round(mean_linear_absolute_error_value,2)}')
print(f'Erreur linéaire absolue moyenne en % = {round(mean_linear_absolute_error_value_percent,2)} %')
print(f'Erreur linéaire quadratique moyenne flat = {round(mean_linear_squared_error_value,2)}')
print(f'Erreur linéaire quadratique moyenne en % = {round(mean_linear_squared_error_value_percent,2)} %')
print('')
print(f'Erreur polynomiale absolue moyenne flat = {round(mean_poly_absolute_error_value,2)}')
print(f'Erreur polynomiale absolue moyenne en % = {round(mean_poly_absolute_error_value_percent,2)} %')
print(f'Erreur polynomiale quadratique moyenne flat = {round(mean_poly_squared_error_value,2)}')
print(f'Erreur polynomiale quadratique moyenne en % = {round(mean_poly_squared_error_value_percent,2)} %')

# Affichage d'un diagrame résiduel pour savoir si les données sont adaptées à de la regression linéaire
# test_residuals = y_test - test_linear_predictions
# sns.scatterplot(x=y_test, y=test_residuals)
# plt.axhline(y=0, color= 'r', ls='--')
# plt.show()

import duckdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Récupération des données historiques dans le fichier parquet
conn = duckdb.connect(database=':memory:', read_only=False)
conn.execute("CREATE TABLE my_table AS SELECT * FROM parquet_scan('data/fake_data.parquet')")
query = "SELECT * FROM my_table LIMIT 5"
df = conn.execute(query).fetch_df()
# print(df.head())

# Décomposition train/test
X = df[['jours_homme_dispo']]
y = df['velocite_reelle']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Entraînement
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
test_predictions = model.predict(X_test)

# Calcul des erreurs
mean_velocity = df['velocite_reelle'].mean()

mean_absolute_error_value = mean_absolute_error(y_test, test_predictions)
mean_absolute_error_value_percent = mean_velocity * mean_absolute_error_value / 100

mean_squared_error_value = np.sqrt(mean_squared_error(y_test, test_predictions))
mean_squared_error_value_percent = mean_velocity * mean_squared_error_value / 100

print(f'Vélocité moyenne = {mean_velocity}')
print(f'Erreur absolue moyenne flat = {round(mean_absolute_error_value,2)}')
print(f'Erreur absolue moyenne en % = {round(mean_absolute_error_value_percent,2)} %')
print(f'Erreur quadratique moyenne flat = {round(mean_squared_error_value,2)}')
print(f'Erreur quadratique moyenne en % = {round(mean_squared_error_value_percent,2)} %')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Cargar datos
df = pd.read_csv("dataset/historico3.csv")
df['fecha'] = pd.to_datetime(df['fecha'])

# Nuevas variables temporales
df['mes'] = df['fecha'].dt.month
df['dia_semana'] = df['fecha'].dt.dayofweek
df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)

# One-hot encoding para 'producto' y 'estacion'
df = pd.get_dummies(df, columns=['producto', 'estacion'], drop_first=True)

# Variables independientes y dependientes
X = df.drop(columns=['fecha', 'ventas'])
y = df['ventas']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo
model = xgb.XGBRegressor(random_state=42)

# Definir los hiperparámetros a buscar
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Realizar la búsqueda en cuadrícula
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)

# Ajustar el modelo a los datos
grid_search.fit(X_train, y_train)

# Mejor combinación de hiperparámetros
best_params = grid_search.best_params_
print(f"Mejores hiperparámetros encontrados: {best_params}")

# Usar el mejor modelo encontrado
best_model = grid_search.best_estimator_

# Guardar el modelo
joblib.dump(best_model, 'modelo_ajustado.pkl')

# Predicción
pred = best_model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

print("Modelo entrenado y guardado exitosamente.")
print(f"Error Absoluto Medio (MAE): {mae}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
print(f"Coeficiente de Determinación (R²): {r2}")

# Validación cruzada
scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"MAE promedio con validación cruzada: {-scores.mean()}")

# Comparar predicciones vs reales
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Ventas reales', linestyle='--')
plt.plot(pred, label='Predicciones', linestyle=':')
plt.title('Comparación entre ventas reales y predichas')
plt.legend()
plt.savefig("ventas_vs_predicciones.png")
plt.close()

# Errores residuales
residuos = y_test - pred
plt.scatter(y_test, residuos, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Ventas reales')
plt.ylabel('Error de predicción')
plt.title('Errores residuales')
plt.savefig("errores_residuales.png")
plt.close()

# Mapa de correlación
correlaciones = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mapa de Correlación")
plt.tight_layout()
plt.savefig("correlacion.png")
plt.close()

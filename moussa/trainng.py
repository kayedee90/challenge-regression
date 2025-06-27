from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Crée et entraîne le modèle
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



# Prédictions
y_pred = rf_model.predict(X_test)

# Graphique des valeurs réelles vs prédites
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6, label='Préditions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Idéal')
plt.xlabel('Vrais prix (€)')
plt.ylabel('Prix prédits (€)')
plt.title('Comparaison entre vrais prix et prédictions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Sauvegarde du modèle
joblib.dump(rf_model, 'random_forest_model.pkl')

# Chargement du modèle
model = joblib.load('random_forest_model.pkl')



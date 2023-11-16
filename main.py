import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Charger les données
df = pd.read_csv('training_data.csv')

# Encoder les niveaux de difficulté (A1, A2, B1, B2, C1, C2)
label_encoder = LabelEncoder()
df['difficulty_encoded'] = label_encoder.fit_transform(df['difficulty'])

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['difficulty_encoded'], test_size=0.2, random_state=42)

# Vectoriser les phrases
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entraîner un modèle de régression linéaire
model = LinearRegression()
model.fit(X_train_tfidf, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test_tfidf)

# Limiter les prédictions à la plage des étiquettes encodées
y_pred_limited = np.clip(y_pred.round(), 0, len(label_encoder.classes_) - 1).astype(int)

# Convertir les prédictions limitées en étiquettes de difficulté
y_pred_labels = label_encoder.inverse_transform(y_pred_limited)

# Comparer les prédictions avec les valeurs réelles
# comparison_df = pd.DataFrame({'Difficulté réelle': label_encoder.inverse_transform(y_test), 'Difficulté prédite': y_pred_labels})
# print(comparison_df)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred_limited)
print(f'MSE: {mse}')

# Assurez-vous que ces étapes ont été exécutées auparavant :
# 1. Entraînement du modèle (model, vectorizer, label_encoder)
# 2. Sauvegarde du modèle, du vectoriseur et de l'encodeur d'étiquettes si nécessaire

# Charger les nouvelles données
new_data = pd.read_csv('unlabelled_test_data.csv')

# Transformer les nouvelles phrases
new_data_tfidf = vectorizer.transform(new_data['sentence'])

# Prédire les difficultés
new_predictions = model.predict(new_data_tfidf)

# Limiter les prédictions à la plage des étiquettes encodées
new_predictions_limited = np.clip(new_predictions.round(), 0, len(label_encoder.classes_) - 1).astype(int)

# Convertir les prédictions en étiquettes de difficulté
new_predictions_labels = label_encoder.inverse_transform(new_predictions_limited)

# Ajouter les prédictions au DataFrame
new_data['difficulty'] = new_predictions_labels

# Sélectionner uniquement les colonnes 'id' et 'predicted_difficulty'
final_result = new_data[['id', 'difficulty']]

# Afficher le résultat final
print(final_result)

final_result.to_csv('labelled_test_data.csv', index=False)
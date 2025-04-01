import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from multiprocessing import Pool, cpu_count

if not os.path.exists('src/classification_results'):
    os.makedirs('src/classification_results')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

def clean_review(review):
    review = re.sub(r'<.*?>|[^\w\s]', ' ', review.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in review.split() if word not in stop_words and word)

# Function for parallel processing
def process_chunk(chunk):
    return chunk.apply(clean_review)

print("Chargement des données...")
df = pd.read_csv('./data/IMDB Dataset.csv')
print(f"Nombre total de critiques: {len(df)}")
print(f"Distribution des sentiments:\n{df['sentiment'].value_counts()}")

print("\nNettoyage des critiques (version optimisée)...")
num_cores = cpu_count()
print(f"Utilisation de {num_cores} cœurs pour le traitement parallèle...")

chunks = np.array_split(df['review'], num_cores)
pool = Pool(num_cores)
results = pool.map(process_chunk, chunks)
pool.close()
pool.join()

df['review_cleaned'] = pd.concat(results)
print("Nettoyage terminé.")

print("\nVectorisation des critiques...")
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(df['review_cleaned'])
print(f"Dimensions des données vectorisées: {X_vectorized.shape}")

print("\nSéparation des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, df['sentiment'], test_size=0.2, random_state=42)
print(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]}")
print(f"Taille de l'ensemble de test: {X_test.shape[0]}")

print("\nEntraînement du modèle de régression logistique...")
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

cm_logistic = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Matrice de confusion - Régression Logistique')
plt.ylabel('Valeur réelle')
plt.xlabel('Valeur prédite')
plt.tight_layout()
plt.savefig('src/classification_results/confusion_matrix_logistic.png')
plt.close()

print("\nEntraînement du modèle SVC...")
svc_model = SVC()
svc_model.fit(X_train, y_train)

y_pred_svc = svc_model.predict(X_test)

print("Support Vector Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

cm_svc = confusion_matrix(y_test, y_pred_svc)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Matrice de confusion - SVC')
plt.ylabel('Valeur réelle')
plt.xlabel('Valeur prédite')
plt.tight_layout()
plt.savefig('src/classification_results/confusion_matrix_svc.png')
plt.close()

models = ['Logistic Regression', 'SVC']
accuracies = [accuracy_score(y_test, y_pred_logistic), accuracy_score(y_test, y_pred_svc)]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Comparaison des performances des modèles')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)  
plt.tight_layout()
plt.savefig('src/classification_results/model_comparison.png')
plt.close()

print("\nLes résultats et graphiques ont été sauvegardés dans le dossier 'classification_results'")
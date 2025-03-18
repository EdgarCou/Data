import pandas as pd
from textblob import TextBlob

# Charger les données
df = pd.read_csv("./IMDB Dataset.csv/IMDB Dataset.csv")

# Vérifier si la colonne 'Sentiment' existe, sinon la créer
if 'Sentiment' not in df.columns:
    df['Sentiment'] = df['review'].apply(lambda x: "positive" if TextBlob(x).sentiment.polarity > 0 else "negative")

# Supprimer les balises <br /><br /> des données
df = df.replace(r'<br /><br />', ' ', regex=True)

# Fonction pour analyser le sentiment
def sentiment_analysis(review):
    score = TextBlob(review).sentiment.polarity  # Analyse du sentiment
    return "positive" if score > 0 else "negative"

# Ajouter une colonne avec les sentiments calculés
df["Sentiment_Calculé"] = df["review"].apply(sentiment_analysis)

# Comparer avec la colonne Sentiment existante
df["Match"] = df["Sentiment"] == df["Sentiment_Calculé"]

# Sauvegarder les résultats dans un nouveau fichier CSV
df.to_csv("IMDB_Reviews_Comparison.csv", index=False, encoding='utf-8')

# Afficher un aperçu du fichier CSV sous forme de tableau
print("Aperçu de la comparaison des sentiments :\n")
# Afficher les lignes où les sentiments ne correspondent pas
mismatched_reviews = df[df["Match"] == False]
print(mismatched_reviews[["review", "Sentiment", "Sentiment_Calculé"]])

# Calculer et afficher le pourcentage de réussite
success_rate = df["Match"].mean() * 100
print(f"Taux de réussite : {success_rate:.2f}%")

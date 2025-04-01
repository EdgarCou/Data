import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from wordcloud import WordCloud
from multiprocessing import Pool, cpu_count
import numpy as np

if not os.path.exists('exploration_results'):
    os.makedirs('exploration_results')

print("Chargement des données...")
df = pd.read_csv('./datasets/IMDB Dataset.csv') 

print("Aperçu du dataset:")
print(df.head())

print("\nTaille du dataset:", df.shape)

print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

print("\nDistribution des sentiments:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Distribution des sentiments')
plt.savefig('exploration_results/sentiment_distribution.png')
plt.close()

def clean_review(review):
    review = re.sub(r'<.*?>|[^\w\s]', ' ', review.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in review.split() if word not in stop_words and word)

def process_chunk(chunk):
    return chunk.apply(clean_review)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

df['review_length'] = df['review'].apply(len)
df['word_count'] = df['review'].apply(lambda x: len(x.split()))

print("\nStatistiques sur la longueur des critiques (caractères):")
print(df['review_length'].describe())

print("\nStatistiques sur le nombre de mots par critique:")
print(df['word_count'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title('Distribution des longueurs des critiques (caractères)')
plt.xlabel('Longueur (caractères)')
plt.savefig('exploration_results/review_length_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], bins=50, kde=True)
plt.title('Distribution du nombre de mots par critique')
plt.xlabel('Nombre de mots')
plt.savefig('exploration_results/word_count_distribution.png')
plt.close()

print("\nLongueur moyenne des critiques par sentiment (caractères):")
print(df.groupby('sentiment')['review_length'].mean())

print("\nNombre moyen de mots par sentiment:")
print(df.groupby('sentiment')['word_count'].mean())

plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='review_length', data=df)
plt.title('Longueur des critiques par sentiment (caractères)')
plt.savefig('exploration_results/review_length_by_sentiment.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='word_count', data=df)
plt.title('Nombre de mots par sentiment')
plt.savefig('exploration_results/word_count_by_sentiment.png')
plt.close()

nltk.download('punkt')
nltk.download('stopwords')

def count_negation_words(text):
    negation_words = ['not', 'no', 'never', 'neither', 'nor', "n't", 'cannot', 'without']
    words = word_tokenize(text.lower())
    return sum(1 for word in words if word in negation_words)

print("\nAnalyse des mots de négation sur un échantillon...")
sample_df = df.sample(1000, random_state=42)
sample_df['negation_count'] = sample_df['review'].apply(count_negation_words)

print("\nMoyenne de mots de négation par sentiment:")
negation_means = sample_df.groupby('sentiment')['negation_count'].mean()
print(negation_means)

plt.figure(figsize=(8, 6))
sns.barplot(x=negation_means.index, y=negation_means.values)
plt.title('Moyenne de mots de négation par sentiment')
plt.ylabel('Nombre moyen de mots de négation')
plt.savefig('exploration_results/negation_words_by_sentiment.png')
plt.close()

def get_top_words(text_series, n=20):
    words = []
    for text in text_series:
        words.extend(text.split())
    return Counter(words).most_common(n)

# Utilisation des critiques nettoyées pour l'analyse des mots fréquents
pos_sample = sample_df[sample_df['sentiment'] == 'positive']['review_cleaned']
neg_sample = sample_df[sample_df['sentiment'] == 'negative']['review_cleaned']

print("\nMots les plus fréquents dans les critiques positives:")
pos_top_words = get_top_words(pos_sample)
print(pos_top_words)

print("\nMots les plus fréquents dans les critiques négatives:")
neg_top_words = get_top_words(neg_sample)
print(neg_top_words)

plt.figure(figsize=(12, 6))
sns.barplot(x=[word[0] for word in pos_top_words[:10]], y=[word[1] for word in pos_top_words[:10]])
plt.title('Top 10 des mots dans les critiques positives')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('exploration_results/top_words_positive.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(x=[word[0] for word in neg_top_words[:10]], y=[word[1] for word in neg_top_words[:10]])
plt.title('Top 10 des mots dans les critiques négatives')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('exploration_results/top_words_negative.png')
plt.close()

def create_wordcloud(text_series, title, filename):
    all_text = ' '.join([text for text in text_series])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'exploration_results/{filename}.png')
    plt.close()

try:
    # Utilisation des critiques nettoyées pour les nuages de mots
    create_wordcloud(pos_sample, 'Nuage de mots - Critiques positives', 'wordcloud_positive')
    create_wordcloud(neg_sample, 'Nuage de mots - Critiques négatives', 'wordcloud_negative')
except ImportError:
    print("La bibliothèque WordCloud n'est pas installée. Exécutez 'pip install wordcloud' pour l'installer.")

print("\nLes résultats et graphiques ont été sauvegardés dans le dossier 'exploration_results'")
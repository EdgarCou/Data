import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv('./IMDB Dataset.csv/IMDB Dataset.csv') 

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
plt.savefig('sentiment_distribution.png')
plt.close()

df['review_length'] = df['review'].apply(len)
print("\nStatistiques sur la longueur des critiques:")
print(df['review_length'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title('Distribution des longueurs des critiques')
plt.xlabel('Longueur (caractères)')
plt.savefig('review_length_distribution.png')
plt.close()

print("\nLongueur moyenne des critiques par sentiment:")
print(df.groupby('sentiment')['review_length'].mean())

plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='review_length', data=df)
plt.title('Longueur des critiques par sentiment')
plt.savefig('review_length_by_sentiment.png')
plt.close()


nltk.download('punkt')
nltk.download('stopwords')

def count_negation_words(text):
    negation_words = ['not', 'no', 'never', 'neither', 'nor', "n't", 'cannot', 'without']
    words = word_tokenize(text.lower())
    return sum(1 for word in words if word in negation_words)

sample_df = df.sample(1000, random_state=42)
sample_df['negation_count'] = sample_df['review'].apply(count_negation_words)

print("\nMoyenne de mots de négation par sentiment:")
print(sample_df.groupby('sentiment')['negation_count'].mean())


def get_top_words(text_series, n=20):
    stop_words = set(stopwords.words('english'))
    words = []
    for text in text_series:
        text = re.sub(r'[^\w\s]', '', text.lower())
        words.extend([word for word in text.split() if word not in stop_words and len(word) > 2])
    return Counter(words).most_common(n)

pos_sample = sample_df[sample_df['sentiment'] == 'positive']['review']
neg_sample = sample_df[sample_df['sentiment'] == 'negative']['review']

print("\nMots les plus fréquents dans les critiques positives:")
print(get_top_words(pos_sample))

print("\nMots les plus fréquents dans les critiques négatives:")
print(get_top_words(neg_sample))
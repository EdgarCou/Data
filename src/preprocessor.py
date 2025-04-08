import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 

class IMDBPreprocessor:
    def __init__(self, output_dir='src/preprocessing_results'):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def clean_review(self, review):
        review = re.sub(r'<.*?>|[^\w\s]', ' ', review.lower())
        stop_words = set(stopwords.words('english'))
        return ' '.join(word for word in review.split() if word not in stop_words and word)
    
    def preprocess(self, df):
        df_copy = df.copy()
        df_copy['review_cleaned'] = df_copy['review'].apply(self.clean_review)
        return df_copy
    
    def vectorize(self, texts, fit=True):
        if fit:
            return self.vectorizer.fit_transform(texts)
        else:
            return self.vectorizer.transform(texts)
    
    def add_features(self, df):
        df['review_length'] = df['review'].apply(len)
        df['word_count'] = df['review'].apply(lambda x: len(x.split()))
        return df

import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class IMDBPreprocessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def clean_review(self, review):
        review = re.sub(r'<.*?>|[^\w\s]', ' ', review.lower())
        stop_words = set(stopwords.words('english'))
        return ' '.join(word for word in review.split() if word not in stop_words and word)
    
    def fit(self, df):
        self.vectorizer.fit(df['review_cleaned'])
        return self
    
    def transform(self, df, test_size=0.2, random_state=42):
        df['review_cleaned'] = df['review'].apply(self.clean_review)
        
        X_vectorized = self.vectorizer.transform(df['review_cleaned'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, df['sentiment'], test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, df, test_size=0.2, random_state=42):
        df['review_cleaned'] = df['review'].apply(self.clean_review)
        
        X_vectorized = self.vectorizer.fit_transform(df['review_cleaned'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, df['sentiment'], test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def add_features(self, df):
        df['review_length'] = df['review'].apply(len)
        df['word_count'] = df['review'].apply(lambda x: len(x.split()))
        return df
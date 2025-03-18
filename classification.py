import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

def clean_review(review):
    review = re.sub(r'<.*?>', '', review)
    review = review.lower()
    review = re.sub(r'[^\w\s]', '', review)
    tokens = nltk.word_tokenize(review)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df = pd.read_csv('./IMDB Dataset.csv/IMDB Dataset.csv')

df['review_cleaned'] = df['review'].apply(clean_review)

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(df['review_cleaned'])

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, df['sentiment'], test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

svc_model = SVC()
svc_model.fit(X_train, y_train)

y_pred_svc = svc_model.predict(X_test)

print("Support Vector Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from src.preprocessor import IMDBPreprocessor
from src.model import SentimentModel

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'review': [
            "This movie was great, I loved it!",
            "Terrible film, waste of time.",
            "Amazing acting and storyline.",
            "Boring plot and bad acting.",
            "Excellent cinematography and direction.",
            "Disappointing ending, wouldn't recommend."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
    })
    
    train_data = data.iloc[:4]
    test_data = data.iloc[4:]
    
    return train_data, test_data

def test_preprocessor_fit(sample_data):
    train_data, _ = sample_data
    
    preprocessor = IMDBPreprocessor()
    preprocessor.fit(train_data)
    
    assert hasattr(preprocessor.vectorizer, 'vocabulary_')
    assert len(preprocessor.vectorizer.vocabulary_) > 0

def test_preprocessor_transform(sample_data):
    train_data, test_data = sample_data
    
    preprocessor = IMDBPreprocessor()
    preprocessor.fit(train_data)
    
    X_test, y_test = preprocessor.transform(test_data)
    
    assert isinstance(X_test, csr_matrix)
    assert X_test.shape[0] == len(test_data)
    assert all(label in ['positive', 'negative'] for label in y_test)

def test_preprocessor_fit_transform(sample_data):
    train_data, _ = sample_data
    
    preprocessor = IMDBPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_data)
    
    assert isinstance(X_train, csr_matrix)
    assert X_train.shape[0] == len(train_data)
    assert len(y_train) == len(train_data)
    assert all(label in ['positive', 'negative'] for label in y_train)

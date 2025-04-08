import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from src.model import SentimentModel

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing the model"""
    # Create a simple sparse matrix for X
    X = csr_matrix(np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0]
    ]))
    
    # Create labels
    y = np.array(['positive', 'negative', 'negative', 'positive', 'negative', 'positive'])
    
    # Split into train and test
    X_train = X[:4]
    y_train = y[:4]
    X_test = X[4:]
    y_test = y[4:]
    
    return X_train, y_train, X_test, y_test

def test_model_initialization():
    model = SentimentModel(model_type='logistic')
    assert model.model_type == 'logistic'
    assert model.model is not None
    
    model = SentimentModel(model_type='svc')
    assert model.model_type == 'svc'
    assert model.model is not None
    
    with pytest.raises(ValueError):
        SentimentModel(model_type='invalid_type')

def test_train(sample_data):
    X_train, y_train, _, _ = sample_data
    
    logistic_model = SentimentModel(model_type='logistic')
    logistic_model.train(X_train, y_train)
    
    svc_model = SentimentModel(model_type='svc')
    svc_model.train(X_train, y_train)
    

def test_predict(sample_data):
    X_train, y_train, X_test, _ = sample_data
    
    model = SentimentModel(model_type='logistic')
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert len(predictions) == X_test.shape[0]
    assert all(pred in ['positive', 'negative'] for pred in predictions)

def test_evaluate(sample_data):
    X_train, y_train, X_test, y_test = sample_data
    
    model = SentimentModel(model_type='logistic')
    model.train(X_train, y_train)
    
    results = model.evaluate(X_test, y_test)
    
    assert 'accuracy' in results
    assert 'confusion_matrix' in results
    assert 'classification_report' in results
    
    assert 0 <= results['accuracy'] <= 1
    
    assert results['confusion_matrix'].shape == (2, 2)
    
    predictions = model.predict(X_test)
    manual_accuracy = accuracy_score(y_test, predictions)
    assert abs(results['accuracy'] - manual_accuracy) < 1e-10

def test_model_persistence(sample_data, tmp_path):
    X_train, y_train, X_test, _ = sample_data
    
    original_model = SentimentModel(model_type='logistic')
    original_model.train(X_train, y_train)
    
    original_predictions = original_model.predict(X_test)
    
    model_path = tmp_path / "model.joblib"
    original_model.save(str(model_path))
    
    loaded_model = SentimentModel.load(str(model_path))
    
    assert loaded_model.model_type == original_model.model_type
    
    loaded_predictions = loaded_model.predict(X_test)
    assert all(loaded_predictions == original_predictions)
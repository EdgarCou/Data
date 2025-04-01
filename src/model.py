from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class SentimentModel:
    def __init__(self, model_type='logistic', **kwargs):
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, **kwargs)
        elif model_type == 'svc':
            if 'kernel' in kwargs:
                self.model = SVC(**kwargs)
            else:
                self.model = LinearSVC(max_iter=1000, dual=False, **kwargs)
        else:
            raise ValueError("model_type doit être 'logistic' ou 'svc'")
        
        self.model_type = model_type
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return results
import pandas as pd
import os
from src.preprocessor import IMDBPreprocessor
from src.model import SentimentModel
from src.visualization import SentimentVisualizer
from sklearn.model_selection import train_test_split

def main():
    if not os.path.exists('src/classification_results'):
        os.makedirs('src/classification_results')
    
    print("Chargement des données...")
    df = pd.read_csv('./data/IMDB Dataset.csv')
    print(f"Nombre total de critiques: {len(df)}")
    print(f"Distribution des sentiments:\n{df['sentiment'].value_counts()}")
    
    print("\nPrétraitement des données...")
    preprocessor = IMDBPreprocessor()
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    print("Prétraitement terminé.")
    
    df = preprocessor.add_features(df)
    df = preprocessor.preprocess(df)
    
    print("\nGénération des nuages de mots...")
    preprocessor.generate_wordcloud(df, sentiment='positive')
    preprocessor.generate_wordcloud(df, sentiment='negative')
    
    print("\nMots les plus fréquents dans les critiques positives:")
    pos_top_words = preprocessor.get_top_words(df, sentiment='positive', n=10)
    print(pos_top_words)
    
    print("\nMots les plus fréquents dans les critiques négatives:")
    neg_top_words = preprocessor.get_top_words(df, sentiment='negative', n=10)
    print(neg_top_words)
    
    print("\nCréation des visualisations...")
    visualizer = SentimentVisualizer()
    visualizer.plot_sentiment_distribution(df)
    visualizer.plot_review_length_distribution(df)
    visualizer.plot_word_count_distribution(df)
    
    print("\nEntraînement du modèle de régression logistique...")
    logistic_model = SentimentModel(model_type='logistic')
    logistic_model.train(X_train, y_train)
    logistic_results = logistic_model.evaluate(X_test, y_test)
    
    print("Logistic Regression:")
    print("Accuracy:", logistic_results['accuracy'])
    print(logistic_results['classification_report'])
    
    print("\nEntraînement du modèle SVC...")
    svc_model = SentimentModel(model_type='svc')
    svc_model.train(X_train, y_train)
    svc_results = svc_model.evaluate(X_test, y_test)
    
    print("Support Vector Classifier:")
    print("Accuracy:", svc_results['accuracy'])
    print(svc_results['classification_report'])
    
    visualizer.plot_confusion_matrix(logistic_results['confusion_matrix'], "Régression Logistique")
    visualizer.plot_confusion_matrix(svc_results['confusion_matrix'], "SVC")
    
    models = ['Logistic Regression', 'SVC']
    accuracies = [logistic_results['accuracy'], svc_results['accuracy']]
    visualizer.plot_model_comparison(models, accuracies)
    
    print("\nLes résultats et graphiques ont été sauvegardés dans les dossiers 'src/visualization_results' et 'src/classification_results'")

if __name__ == "__main__":
    main()
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.preprocessor import IMDBPreprocessor
from src.model import SentimentModel
from src.visualization import SentimentVisualizer

def main():
    if not os.path.exists('src/visualization_results'):
        os.makedirs('src/visualization_results')
    
    print("Chargement des données...")
    df = pd.read_csv('./data/IMDB Dataset.csv')
    print(f"Nombre total de critiques: {len(df)}")
    print(f"Distribution des sentiments:\n{df['sentiment'].value_counts()}")
    
    print("\nPrétraitement des données...")
    preprocessor = IMDBPreprocessor()
    
    df = preprocessor.preprocess(df)
    print("Nettoyage des critiques terminé.")
    
    df = preprocessor.add_features(df)
    
    print("Vectorisation des critiques...")
    X_vectorized = preprocessor.vectorize(df['review_cleaned'])
    print(f"Dimensions des données vectorisées: {X_vectorized.shape}")
    
    print("\nSéparation des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, df['sentiment'], test_size=0.2, random_state=42
    )
    print(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test: {X_test.shape[0]}")
    
    print("\nCréation des visualisations...")
    visualizer = SentimentVisualizer()
    visualizer.plot_sentiment_distribution(df)
    visualizer.plot_review_length_distribution(df)
    visualizer.plot_word_count_distribution(df)
    
    print("\nCréation des nuages de mots...")
    visualizer.create_sentiment_wordclouds(df, column='review_cleaned')
    
    visualizer.create_colored_wordcloud(
        df['review_cleaned'], 
        'Nuage de mots coloré - Toutes les critiques', 
        'wordcloud_colored',
        colormap='viridis'
    )
    
    visualizer.create_comparative_wordcloud(df, column='review_cleaned')
    print("Nuages de mots créés avec succès.")
    
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
    
    print("\nLes résultats et graphiques ont été sauvegardés dans les dossiers 'src/visualization_results'")

if __name__ == "__main__":
    main()
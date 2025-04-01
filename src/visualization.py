import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.colors as mcolors

class SentimentVisualizer:
    def __init__(self, output_dir='src/visualization_results'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_sentiment_distribution(self, df):
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment', data=df)
        plt.title('Distribution des sentiments')
        plt.savefig(f'{self.output_dir}/sentiment_distribution.png')
        plt.close()
    
    def plot_review_length_distribution(self, df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['review_length'], bins=50, kde=True)
        plt.title('Distribution des longueurs des critiques (caractères)')
        plt.xlabel('Longueur (caractères)')
        plt.savefig(f'{self.output_dir}/review_length_distribution.png')
        plt.close()
    
    def plot_word_count_distribution(self, df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['word_count'], bins=50, kde=True)
        plt.title('Distribution du nombre de mots par critique')
        plt.xlabel('Nombre de mots')
        plt.savefig(f'{self.output_dir}/word_count_distribution.png')
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Matrice de confusion - {model_name}')
        plt.ylabel('Valeur réelle')
        plt.xlabel('Valeur prédite')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_model_comparison(self, models, accuracies):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=accuracies)
        plt.title('Comparaison des performances des modèles')
        plt.ylabel('Accuracy')
        plt.ylim(0.8, 1.0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png')
        plt.close()
    
    def create_wordcloud(self, text_series, title, filename):
        all_text = ' '.join([text for text in text_series])
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png')
        plt.close()
    
    def create_sentiment_wordclouds(self, df, column='review_cleaned'):
        positive_reviews = df[df['sentiment'] == 'positive'][column]
        self.create_wordcloud(
            positive_reviews, 
            'Nuage de mots - Critiques positives', 
            'wordcloud_positive'
        )
        
        negative_reviews = df[df['sentiment'] == 'negative'][column]
        self.create_wordcloud(
            negative_reviews, 
            'Nuage de mots - Critiques négatives', 
            'wordcloud_negative'
        )
    
    def create_colored_wordcloud(self, text_series, title, filename, colormap='viridis'):
        all_text = ' '.join([text for text in text_series])
        
        colormap = plt.cm.get_cmap(colormap)
        colors = [mcolors.rgb2hex(colormap(i)) for i in np.linspace(0, 1, 20)]
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            max_words=100,
            colormap=colormap,
            color_func=lambda *args, **kwargs: np.random.choice(colors)
        ).generate(all_text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png')
        plt.close()
    
    def create_comparative_wordcloud(self, df, column='review_cleaned'):
        positive_reviews = df[df['sentiment'] == 'positive'][column]
        negative_reviews = df[df['sentiment'] == 'negative'][column]
        
        positive_text = ' '.join([text for text in positive_reviews])
        negative_text = ' '.join([text for text in negative_reviews])
        
        positive_cloud = WordCloud(
            width=800, height=400, background_color='white', 
            max_words=100, colormap='Blues'
        ).generate(positive_text)
        
        negative_cloud = WordCloud(
            width=800, height=400, background_color='white', 
            max_words=100, colormap='Reds'
        ).generate(negative_text)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(positive_cloud, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('Critiques positives', fontsize=20)
        
        ax2.imshow(negative_cloud, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('Critiques négatives', fontsize=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/wordcloud_comparison.png')
        plt.close()
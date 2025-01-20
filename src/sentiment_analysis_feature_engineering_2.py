import numpy as np
from tqdm import tqdm
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class SentimentFeatureEngineering:
    
    def __init__(self, texts: list, max_features: int = None, ngram_range: tuple = (1, 1)) -> None:
        """
        Initialize SentimentFeatureEngineering with texts and parameters
        
        Arguments:
        ----------
            texts        : List of preprocessed text documents
            max_features : Maximum number of features (None for no limit)
            ngram_range  : Range of n-grams to consider (min_n, max_n)
            
        Raises:
        -------
            ValueError   : If texts is empty or parameters are invalid
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")
        self.texts = texts
        self.max_features = max_features
        self.ngram_range = ngram_range

    def polarity_scores(self, text: str) -> tuple:
        """
        Extract polarity scores using TextBlob and Vader
        
        Arguments:
        ----------
            text { str } : Text document to analyze
        
        Returns:
        --------
            { tuple } : Tuple containing:
                        - Polarity score (TextBlob)
                        - Subjectivity score (TextBlob)
                        - Vader positive, neutral, negative scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            vader_analyzer = VaderAnalyzer()
            vader_scores = vader_analyzer.polarity_scores(text)
            return polarity, subjectivity, vader_scores['pos'], vader_scores['neu'], vader_scores['neg']
        except Exception as e:
            raise

    def emotion_dictionary_features(self, text: str) -> tuple:
        """
        Extract emotion dictionary features using Vader
        
        Arguments:
        ----------
            text { str } : Text document to analyze
        
        Returns:
        --------
            { tuple } : Tuple containing:
                        - Positive emotion score
                        - Negative emotion score
                        - Neutral emotion score
                        - Compound emotion score
        """
        try:
            sia = VaderAnalyzer()
            sentiment_scores = sia.polarity_scores(text)
            return sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['compound']
        except Exception as e:
            raise

    def aspect_based_features(self, text: str, aspects: list = ['acting', 'plot', 'directing', 'cinematography']) -> dict:
        """
        Extract aspect-based features from text based on predefined aspect keywords
        
        Arguments:
        ----------
            text    { str }  : Text document to analyze
            aspects { list } : List of aspects to count in the text
        
        Returns:
        --------
            { dict } : Dictionary mapping aspect names to counts in the text
        """
        try:
            aspect_counts = {aspect: text.lower().count(aspect) for aspect in aspects}
            return aspect_counts
        except Exception as e:
            raise

    def create_all_features(self) -> dict:
        """
        Create all available sentiment-related features
        
        Returns:
        --------
            { dict } : Dictionary mapping feature names to their feature matrices
        """
        try:
            features = dict()
            features['polarity_scores'] = [self.polarity_scores(text) for text in tqdm(self.texts, desc='Polarity Scores')]
            features['emotion_dictionary'] = [self.emotion_dictionary_features(text) for text in tqdm(self.texts, desc='Emotion Dictionary')]
            features['aspect_features'] = [self.aspect_based_features(text) for text in tqdm(self.texts, desc='Aspect Features')]
            return features
        except Exception as e:
            raise

    def create_tfidf_features(self) -> np.ndarray:
        """
        Create TF-IDF features for the texts
        
        Returns:
        --------
            { np.ndarray } : TF-IDF feature matrix
        """
        try:
            vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
            tqdm_iterator = tqdm(self.texts, desc="Fitting TF-IDF", total=len(self.texts))
            X = vectorizer.fit_transform(tqdm_iterator)
            return X.toarray()
        except Exception as e:
            raise

    def prepare_data_for_training(self) -> tuple:
        """
        Prepare the features and labels for training
        
        Returns:
        --------
            { tuple } : Tuple containing:
                        - Feature matrix (X)
                        - Label vector (y)
        """
        try:
            features = self.create_all_features()
            X = np.hstack([
                np.array(features['polarity_scores']),
                np.array(features['emotion_dictionary']),
                np.array([list(aspect.values()) for aspect in features['aspect_features']]),
                self.create_tfidf_features()
            ])
            y = np.array([1 if 'positive' in text else 0 for text in self.texts])
            return X, y
        except Exception as e:
            raise

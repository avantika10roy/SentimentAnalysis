import numpy as np
import pandas as pd
import spacy
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict


class SentimentFeatureEngineering:
    """
    A class for implementing various sentiment feature engineering techniques
    
    Attributes:
    -----------
        texts           { list }  : List of preprocessed text documents
        
        nrc_path        { str }   : Path to NRC emotion lexicon file
        
        batch_size      { int }   : Size of batches for processing
    """
    
    def __init__(self, texts: list, nrc_path: str = "data/emotion_lexicon/NRC-Emotion-Lexicon-v0.92.txt", 
                 batch_size: int = 1000) -> None:
        """
        Initialize SentimentFeatureEngineering with texts and parameters
        
        Arguments:
        ----------
            texts      : List of preprocessed text documents
            
            nrc_path   : Path to NRC emotion lexicon file
            
            batch_size : Size of batches for processing
            
        Raises:
        -------
            ValueError : If texts is empty or parameters are invalid
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")
            
        self.texts = texts
        self.batch_size = batch_size
        self.feature_names = []
        
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        self.vader = SentimentIntensityAnalyzer()
        self._load_nrc_lexicon(nrc_path)
        
    def _load_nrc_lexicon(self, path: str) -> None:
        """
        Load and process NRC emotion lexicon
        """
        try:
            nrc_df = pd.read_csv(path, sep='\t', header=None, 
                                names=['word', 'emotion', 'association'])
            nrc_filtered = nrc_df[nrc_df['association'] == 1]
            
            self.emotion_dict = defaultdict(list)
            for _, row in nrc_filtered.groupby('word')['emotion'].apply(list).items():
                self.emotion_dict[_] = row
                
        except Exception as e:
            raise

    class SentimentTransformer(BaseEstimator, TransformerMixin):
        """
        Transformer class for extracting comprehensive sentiment features
        """
        def __init__(self, analyzer, batch_size=1000):
            self.analyzer = analyzer
            self.batch_size = batch_size
            self.feature_names = []

        def fit(self, texts, y=None):
            return self

        def transform(self, texts):
            results = self.analyzer.batch_analyze_sentiment(texts, self.batch_size)
            feature_dict = defaultdict(list)

            # Process each result and extract features
            for result in results:
                # VADER features
                vader = result['vader_sentiment']
                for k, v in vader.items():
                    feature_dict[f'vader_{k}'].append(v)

                # Emotion counts
                emotions = result['emotion_counts']
                for emotion, count in emotions.items():
                    feature_dict[f'emotion_{emotion}'].append(count)

                # Aspect sentiments
                aspects = result['aspect_based_sentiment']
                if aspects:
                    sentiments = list(aspects.values())
                    feature_dict['aspect_mean'].append(np.mean(sentiments))
                    feature_dict['aspect_min'].append(min(sentiments))
                    feature_dict['aspect_max'].append(max(sentiments))
                else:
                    feature_dict['aspect_mean'].append(0)
                    feature_dict['aspect_min'].append(0)
                    feature_dict['aspect_max'].append(0)

                # Polarity patterns
                polarity = result['polarity_patterns']
                for k, v in polarity.items():
                    if k != 'overall_sentiment':
                        feature_dict[f'polarity_{k}'].append(v)

            # Convert to sparse matrix
            self.feature_names = list(feature_dict.keys())
            rows = len(results)
            cols = len(self.feature_names)
            data = []
            row_ind = []
            col_ind = []

            for col_idx, feature in enumerate(self.feature_names):
                for row_idx, value in enumerate(feature_dict[feature]):
                    if value != 0:
                        data.append(value)
                        row_ind.append(row_idx)
                        col_ind.append(col_idx)

            return csr_matrix((data, (row_ind, col_ind)), shape=(rows, cols))

        def get_feature_names(self):
            return self.feature_names

    def create_comprehensive_features(self) -> tuple:
        """
        Create comprehensive sentiment features including VADER, emotions, aspects, and polarity
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted SentimentTransformer
                                        - Comprehensive sentiment feature matrix
        """
        try:
            print("Creating comprehensive sentiment features...")
            transformer = self.SentimentTransformer(self, self.batch_size)
            features = transformer.fit_transform(self.texts)
            
            print(f"Created {features.shape[1]} comprehensive sentiment features")
            return transformer, features
            
        except Exception as e:
            raise

    def create_all_sentiment_features(self) -> dict:
        """
        Create all available sentiment feature types
        
        Returns:
        --------
            { dict } : Dictionary mapping feature names to their transformer and feature matrix
        """
        try:
            print("Creating all sentiment feature types...")
            features = dict()
            
            # Create comprehensive features
            features['comprehensive'] = self.create_comprehensive_features()
            
            print("Created all sentiment feature types successfully")
            return features
            
        except Exception as e:
            raise

    def get_feature_names(self):
        """
        Returns the names of all extracted features
        """
        return self.feature_names
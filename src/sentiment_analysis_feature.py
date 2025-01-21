import numpy as np
import pandas as pd
import spacy
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from config import Emotion_path


class SentimentFeatureEngineering:
    """
    A class for implementing various sentiment feature engineering techniques
    
    Attributes:
    -----------
        texts           { list }  : List of preprocessed text documents
        nrc_path        { str }   : Path to NRC emotion lexicon file
        batch_size      { int }   : Size of batches for processing
    """
    
    def __init__(self, texts: list, nrc_path = Emotion_path, 
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

    def batch_analyze_sentiment(self, texts: list, batch_size: int) -> list:
        """
        Analyze sentiment for a batch of texts
        
        Arguments:
        ----------
        texts: List of text documents
        batch_size: Number of documents to process at once
        
        Returns:
        --------
        List of dictionaries containing sentiment analysis results
        """
        results = []
        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = []
                
                # Process each text in the batch
                docs = list(self.nlp.pipe(batch))
                vader_sentiments = [self.vader.polarity_scores(text) for text in batch]
                
                for doc, vader_sent in zip(docs, vader_sentiments):
                    result = {}
                    
                    # VADER sentiment
                    result['vader_sentiment'] = {
                        'positive': vader_sent['pos'],
                        'negative': vader_sent['neg'],
                        'neutral': vader_sent['neu'],
                        'compound': vader_sent['compound']
                    }
                    
                    # Emotion counts
                    emotion_counts = defaultdict(int)
                    for token in doc:
                        word = token.text.lower()
                        if word in self.emotion_dict:
                            for emotion in self.emotion_dict[word]:
                                emotion_counts[emotion] += 1
                    result['emotion_counts'] = dict(emotion_counts)
                    
                    # Aspect-based sentiment
                    aspects = {}
                    for token in doc:
                        if token.pos_ == "NOUN":
                            aspect_text = token.text
                            aspects[aspect_text] = self.vader.polarity_scores(aspect_text)['compound']
                    result['aspect_based_sentiment'] = aspects
                    
                    # Polarity patterns
                    result['polarity_patterns'] = {
                        'positive': vader_sent['pos'],
                        'negative': vader_sent['neg'],
                        'neutral': vader_sent['neu']
                    }
                    
                    batch_results.append(result)
                
                results.extend(batch_results)
            
        except Exception as e:
            print(f"Error during sentiment analysis: {str(e)}")
            raise
        
        return results
            
    def _load_nrc_lexicon(self, path: str) -> None:
        """
        Load and process NRC emotion lexicon
        """
        try:
            nrc_df = pd.read_csv(path, sep='\t', header=None, 
                                  names=['word', 'emotion', 'association'])
            nrc_filtered = nrc_df[nrc_df['association'] == 1]
            
            # Create the emotion dictionary
            self.emotion_dict = defaultdict(list)
            for _, row in nrc_filtered.groupby('word')['emotion'].apply(list).items():
                self.emotion_dict[_] = row
                
        except Exception as e:
            print(f"Error loading NRC lexicon: {str(e)}")
            raise

    class SentimentTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, analyzer, batch_size=1000):
            self.analyzer = analyzer
            self.batch_size = batch_size
            self.feature_names = []
            self.fitted = False
            self.emotion_set = set()  

        def fit(self, texts, y=None):
            """
            Fit the transformer by analyzing a sample of texts and establishing feature names
            """
            # Get initial results to determine feature structure
            results = self.analyzer.batch_analyze_sentiment(texts, self.batch_size)
            if results:
                # Get sample result for structure
                sample_result = results[0]
                
                # First collect all unique emotions across all results
                self.emotion_set = set()
                for result in results:
                    self.emotion_set.update(result['emotion_counts'].keys())
                
                # Now build feature names using the complete emotion set
                self.feature_names = (
                    # VADER sentiment features
                    [f'vader_{k}' for k in sample_result['vader_sentiment'].keys()] +
                    # Emotion features using the collected set
                    [f'emotion_{emotion}' for emotion in sorted(self.emotion_set)] +
                    # Aspect-based sentiment features
                    ['aspect_mean', 'aspect_min', 'aspect_max'] +
                    # Polarity pattern features
                    [f'polarity_{k}' for k in sample_result['polarity_patterns'].keys() 
                    if k != 'overall_sentiment']
                )
                
            self.fitted = True
            return self

        def transform(self, texts):
            """
            Transform texts into feature matrix using the established feature names
            """
            if not self.fitted:
                raise ValueError("Transformer must be fitted before transform")
            
            results = self.analyzer.batch_analyze_sentiment(texts, self.batch_size)
            feature_dict = defaultdict(list)

            # Process each result and extract features
            for result in results:
                # VADER features
                vader = result['vader_sentiment']
                for k, v in vader.items():
                    feature_dict[f'vader_{k}'].append(v)

                # Emotion counts - use the established emotion set
                emotions = result['emotion_counts']
                for emotion in sorted(self.emotion_set):  # Use sorted to maintain order
                    feature_dict[f'emotion_{emotion}'].append(emotions.get(emotion, 0))

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
            if not self.fitted:
                raise ValueError("Transformer must be fitted before getting feature names")
            return self.feature_names
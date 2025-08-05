# Dependencies
import nltk
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ignore all runtime warnings
warnings.filterwarnings('ignore')

class Contextual_Features:
    """
    A class for implementing various text feature engineering techniques
    
    Attributes:
    -----------
        texts        { list }  : List of preprocessed text documents
        
        max_features  { int }  : Maximum number of features to create
        
        ngram_range  { tuple } : Range of n-grams to consider
    """
    
    def __init__(self, texts: list, max_features: int = None, ngram_range: tuple = (1, 3)) -> None:
        """
        Initialize TextFeatureEngineering with texts and parameters
        
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
            
        self.texts        = texts
        self.max_features = max_features
        self.ngram_range  = ngram_range
        
    def window_based(self):
        """
        Create Window Based Feature Engineering with texts and parameters

        Arguments:
        ----------
        texts             : List of preprocessed text documents
        """
        try:
            print("Creating Window-Based Contextual Features:...")
            vectorizer = CountVectorizer(max_features = self.max_features,
                                         ngram_range  = self.ngram_range)
            ngrams_features     = vectorizer.fit_transform(self.texts)
            
            return vectorizer, ngrams_features
            
        except Exception as e:
            raise

    '''def position_based(self):
        """
        Create Position Based Feature Engineering with texts and parameters

        Arguments:
        ----------
        texts              : List of preprocessed text documents
        """
        try:
            print("Creating Position-Based Contextual Features:...")
            position_features = []
            
            position_vectorizer = CountVectorizer(max_features = self.max_features,
                                                  ngram_range  = self.ngram_range)

            for doc in self.texts:
                words = doc.split() 
            
                position_features.extend([{"word": word, "position": idx} for idx, word in enumerate(words)])

            return position_vectorizer, position_features

        except Exception as e:
            raise'''
    
    def position_based(self):
        """
        Create Position Based Feature Engineering with texts and parameters

        Arguments:
        ----------
        texts              : List of preprocessed text documents
        """
        try:
            print("Creating Position-Based Contextual Features:...")
            position_vectorizer = CountVectorizer(max_features = self.max_features,
                                                  ngram_range  = self.ngram_range)

            position_features = position_vectorizer.fit_transform(self.texts)
            
            return position_vectorizer, position_features

        except Exception as e:
            raise

    '''def generate_ngrams(self, n=3):
        """
        Generate N-Grams

        Arguments:
        ----------
        words         : List of words taken individually from the preprocessed text documents
        n             : Individual words from the list
        """
        print("Generating N-Grams:...")
        ngrams = []
        
        ngrams_vectorizer = CountVectorizer(max_features = self.max_features,
                                            ngram_range  = self.ngram_range)

        for doc in self.texts:
            words = doc.split() 
            ngrams.extend([tuple(words[i:i+n]) for i in range(len(words)-n+1)]) 

        return ngrams_vectorizer, ngrams'''
    
    def generate_ngrams(self, n=3):
        """
        Generate N-Grams

        Arguments:
        ----------
        words         : List of words taken individually from the preprocessed text documents
        n             : Individual words from the list
        """
        print("Generating N-Grams:...")
        
        ngrams_vectorizer = CountVectorizer(max_features = self.max_features,
                                            ngram_range  = self.ngram_range)

        ngrams_features = ngrams_vectorizer.fit_transform(self.texts)

        return ngrams_vectorizer, ngrams_features


    def cross_document(self):
        """
        Create Cross Document Feature Engineering with texts and parameters

        Arguments:
        ----------
        texts             : List of preprocessed text documents
        """
        try: 
            print("Creating Cross Document Contextual Feature Engineering:...")
            vectorizer   = TfidfVectorizer(max_features = self.max_features, 
                                           ngram_range  = self.ngram_range)
            tfidf_matrix = vectorizer.fit_transform(self.texts)
            return vectorizer, tfidf_matrix

        except Exception as e:
            raise

import textstat
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

class Statistical_Feature_Engineering():
    """
    A class for statistical feature engineering.

    """
    def __init__(self,max_features=1000):
        """
        Intializes the Statistical_Feature_Engineering.
        """
        self.max_features = max_features

    def document_statistics(self, text, cleaned_text):
        """
        Calculates basic documnet statistics i.e. Character Count, Word Count, Sentence Count, Average Word Length(AWL), 
        Average Sentence Length(ASL), Unique Word Ratio(UWR).

        Arguments:
        ----------
        text {list} : List of uncleaned texts.
        cleaned_text {list} : List of cleaned texts.

        Returns:
        --------
        stats_matrix {csr_matrix} : Output data with the calculated document statistics.
        
        """
        
        char_count = np.array([len(doc) for doc in cleaned_text])
        word_count = np.array([len(doc.split()) for doc in cleaned_text])
        sent_count = np.array([len(sent_tokenize(doc)) for doc in text])
        AWL = char_count/word_count
        ASL = word_count/sent_count
        unique_word_count = np.array([len(set(word_tokenize(doc))) for doc in cleaned_text])
        UWR = unique_word_count/word_count

        stats_matrix = csr_matrix(np.column_stack((char_count, word_count, sent_count, AWL, ASL, UWR)))
        return stats_matrix


    def readability_score(self, cleaned_text, score='FRE'):
        """
        Calculates the readability scores i.e. Flesch Readine Ease(FRE), Gunning Fog Index(GFI), SMOG Index(SMOG) 
        and ALL(for all types of score).

        Arguments:
        ----------
        cleaned_text {list} : List of cleaned texts.

        score {str} : score type {'FRE', 'GFI', 'SMOG', 'ALL'}.

        Returns:
        ---------
        {csr_matrix}: readability_score according to the score type.
        
        """
        if score == 'FRE':
            scores = [textstat.flesch_reading_ease(doc) for doc in cleaned_text]
            return csr_matrix(scores.values.reshape(-1, 1))
        elif score == 'GFI':
            scores = [textstat.gunning_fog(doc) for doc in cleaned_text]
            return csr_matrix(scores.values.reshape(-1, 1))
        elif score == 'SMOG':
            scores = [textstat.smog_index(doc) for doc in cleaned_text]
            return csr_matrix(scores.values.reshape(-1, 1))
        elif score == 'ALL':
            fre = [textstat.flesch_reading_ease(doc) for doc in cleaned_text]
            gfi = [textstat.gunning_fog(doc) for doc in cleaned_text]
            smog = [textstat.smog_index(doc) for doc in cleaned_text]
            
            combined_scores = pd.DataFrame({'FRE': fre, 'GFI': gfi, 'SMOG': smog})
            return csr_matrix(combined_scores.values)
        else:
            raise ValueError("Unsupported score type. Choose from 'FRE', 'GFI', 'SMOG', or 'ALL'.")

        

    def frequency_distribution(self,cleaned_text):
        """
        Calculates the word counts in each document.

        Arguments:
        ----------

        cleaned_text {list} : Input list of cleaned texts.

        column {str} : Column name for calculating the frequency distribution.

        Returns:
        -------
        X {csr_matrix} : vectorized sparse matrix.
        
        """
        vectorizer = CountVectorizer(max_features=self.max_features)
        X = vectorizer.fit_transform(cleaned_text)
        return vectorizer, X
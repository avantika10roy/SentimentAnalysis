# Deoendencies
import textstat
import numpy as np
import pandas as pd
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

    def create_document_statistics(self, text, cleaned_text):
        """
        Calculates basic documnet statistics i.e. Character Count, Word Count, Sentence Count, Average Word Length(AWL), 
        Average Sentence Length(ASL), Unique Word Ratio(UWR).

        Arguments:
        ----------
        text {list} : List of uncleaned texts.
        cleaned_text {list} : List of cleaned texts.

        Returns:
        --------
        doct_vect {object} : vectorizer object. 

        doc_stat {csr_matrix} : Output data with the calculated document statistics.
        
        """
        if not text:
            raise ValueError("text can not be empty.")
        
        if not cleaned_text:
            raise ValueError("cleaned_text can not be empty.")
        
        try:
            class document_statistics():
                def __init__(self):
                    pass

                def fit_transform(self, text, cleaned_text):
                    char_count = np.array([len(doc) for doc in cleaned_text])
                    word_count = np.array([len(doc.split()) for doc in cleaned_text])
                    sent_count = np.array([len(sent_tokenize(doc)) for doc in text])
                    AWL = char_count/word_count
                    ASL = word_count/sent_count
                    unique_word_count = np.array([len(set(word_tokenize(doc))) for doc in cleaned_text])
                    UWR = unique_word_count/word_count

                    stats_matrix = csr_matrix(np.column_stack((char_count, word_count, sent_count, AWL, ASL, UWR)))
                    return stats_matrix
                
                def transform(self, text, cleaned_text):
                    char_count = np.array([len(doc) for doc in cleaned_text])
                    word_count = np.array([len(doc.split()) for doc in cleaned_text])
                    sent_count = np.array([len(sent_tokenize(doc)) for doc in text])
                    AWL = char_count/word_count
                    ASL = word_count/sent_count
                    unique_word_count = np.array([len(set(word_tokenize(doc))) for doc in cleaned_text])
                    UWR = unique_word_count/word_count

                    stats_matrix = csr_matrix(np.column_stack((char_count, word_count, sent_count, AWL, ASL, UWR)))
                    return stats_matrix
            
            doct_vect = document_statistics()
            doc_stat = doct_vect.fit_transform(text=text, cleaned_text=cleaned_text)
            return doct_vect, doc_stat
        
        except Exception as e:
            raise


    def create_readability_score(self, cleaned_text, score='ALL'):
        """
        Calculates the readability scores i.e. Flesch Readine Ease(FRE), Gunning Fog Index(GFI), SMOG Index(SMOG) 
        and ALL(for all types of score).

        Arguments:
        ----------
        cleaned_text {list} : List of cleaned texts.

        score {str} : score type {'FRE', 'GFI', 'SMOG', 'ALL'}.

        Returns:
        ---------
        readability_vectorizer {object} : vectorizer object.
        readability_features {csr_matrix} : readability_score according to the score type.
        
        """
        if not cleaned_text:
            raise ValueError("cleaned_text can not be empty.")
        
        if score not in {'FRE', 'GFI', 'SMOG', 'ALL'}:
            raise ValueError("Unsupported score type. Choose from 'FRE', 'GFI', 'SMOG', or 'ALL'.")

        try:

            class readability_score():
                def __init__(self):
                    pass

                def fit_transform(self, text, score='ALL'):
                    if score == 'FRE':
                        scores = [textstat.flesch_reading_ease(doc) for doc in text]
                        
                    elif score == 'GFI':
                        scores = [textstat.gunning_fog(doc) for doc in text]

                    elif score == 'SMOG':
                        scores = [textstat.smog_index(doc) for doc in text]

                    elif score == 'ALL':
                        fre = [textstat.flesch_reading_ease(doc) for doc in text]
                        gfi = [textstat.gunning_fog(doc) for doc in text]
                        smog = [textstat.smog_index(doc) for doc in text]
                        combined_scores = pd.DataFrame({'FRE': fre, 'GFI': gfi, 'SMOG': smog})
                        return csr_matrix(combined_scores.values)
                    
                    return csr_matrix(scores.values.reshape(-1, 1))

                def transform(self, text, score='ALL'):
                    if score == 'FRE':
                        scores = [textstat.flesch_reading_ease(doc) for doc in text]
                        
                    elif score == 'GFI':
                        scores = [textstat.gunning_fog(doc) for doc in text]

                    elif score == 'SMOG':
                        scores = [textstat.smog_index(doc) for doc in text]

                    elif score == 'ALL':
                        fre = [textstat.flesch_reading_ease(doc) for doc in text]
                        gfi = [textstat.gunning_fog(doc) for doc in text]
                        smog = [textstat.smog_index(doc) for doc in text]
                        combined_scores = pd.DataFrame({'FRE': fre, 'GFI': gfi, 'SMOG': smog})
                        return csr_matrix(combined_scores.values)
                    
                    return csr_matrix(scores.values.reshape(-1, 1))
                    
            readability_vectorizer = readability_score()
            readability_features = readability_vectorizer.fit_transform(text=cleaned_text, score=score)
            return readability_vectorizer, readability_features
        
        except Exception as e:
            raise
        

    def create_frequency_distribution(self,cleaned_text):
        """
        Calculates the word counts in each document.

        Arguments:
        ----------

        cleaned_text {list} : Input list of cleaned texts.

        column {str} : Column name for calculating the frequency distribution.

        Returns:
        -------
        count_vectorizer {object} : vectorizer object.

        X {csr_matrix} : vectorized sparse matrix.
        
        """
        if not cleaned_text:
            raise ValueError("cleaned_text can not be empty.")
        try:
            count_vectorizer = CountVectorizer(max_features=self.max_features)
            bow = count_vectorizer.fit_transform(cleaned_text)
            return count_vectorizer, bow
        
        except Exception as e:
            raise
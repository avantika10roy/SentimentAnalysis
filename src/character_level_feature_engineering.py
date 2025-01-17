from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import string
import pandas as pd

# character level feature engineering for imdb sentiment analysis
class CharacterLevelFeatureEngineering:
    """
    A class for implementing various character level text feature engineering techniques
    
    Attributes:
    -----------
        texts        { list }  : List of preprocessed text documents
        
        max_features  { int }  : Maximum number of features to create
        
    """

    def __init__(self, texts : list, max_features: int = 2000, ngram_range: tuple = (1, 3)) -> None:
        """
        Initialize CharacterLevelFeatureEngineering with texts and parameters
        
        Arguments:
        ----------
            texts        : List of preprocessed text documents
            
            max_features : Maximum number of features (default : 500)

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

    
    def create_char_level_ngram_binary(self) -> tuple:
        """
        Create binary character-level n-gram features (presence/absence).
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted CountVectorizer
                                        - Binary document-term matrix
        """
        try:
            print("Creating binary character-level n-gram features...")
            vectorizer = CountVectorizer(analyzer='char', 
                                        ngram_range=self.ngram_range, 
                                        max_features=self.max_features, 
                                        binary=True)
            
            features = vectorizer.fit_transform(self.texts)
            print(f"Created {features.shape[1]} binary features")
            
            return vectorizer, features
            
        except Exception as e:
            print(f"Error: {e}")
            raise



    def create_char_level_frequency(self) -> tuple:
        """
        Create frequency-based character-level n-gram features.
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted CountVectorizer
                                        - Frequency document-term matrix
        """
        try:
            print("Creating frequency-based character-level n-gram features...")
            vectorizer = CountVectorizer(analyzer='char', 
                                        ngram_range=self.ngram_range, 
                                        max_features=self.max_features)
            
            features = vectorizer.fit_transform(self.texts)
            print(f"Created {features.shape[1]} frequency features")
            
            return vectorizer, features
            
        except Exception as e:
            print(f"Error: {e}")
            raise



    def create_word_length_patterns(self, in_text: str = None) -> np.ndarray:
        """
        Create character-level features based on word length patterns.
        
        Returns:
        --------
            { np.ndarray } : A NumPy array where each row represents a document
                            and each column represents a feature related to word lengths.
                            Features include: 
                            - Average word length
                            - Standard deviation of word lengths
                            - Median word length
                            - Maximum word length
                            - Minimum word length
        """
        try:
            print("Creating word length pattern features...")
            
            features = []


            # Check if in_text is provided, otherwise use the default texts
            text_list = in_text if in_text is not None else self.texts

            if isinstance(text_list, pd.Series):
                text_list = text_list.tolist()  # Convert pandas Series to list if needed

            for text in text_list:
                # Split text into words and compute word lengths
                word_lengths = [len(word) for word in text.split() if word.isalpha()]
                
                # If no valid words, default to zeros
                if len(word_lengths) == 0:
                    features.append([0, 0, 0, 0, 0])
                    continue
                
                # Compute statistical features
                avg_word_length = np.mean(word_lengths)
                std_word_length = np.std(word_lengths)
                median_word_length = np.median(word_lengths)
                max_word_length = np.max(word_lengths)
                min_word_length = np.min(word_lengths)
                
                # Append to feature list
                features.append([
                    avg_word_length,
                    std_word_length,
                    median_word_length,
                    max_word_length,
                    min_word_length
                ])
            
            print(f"Extracted word length patterns for {len(features)} documents.")
            return np.array(features)
        
        except Exception as e:
            print(f"Error: {e}")
            raise



    def create_character_type_ratios(self, in_text: str = None) -> np.ndarray:
        """
        Create character-level features based on the ratios of letters, digits, and special characters.
        
        Returns:
        --------
            { np.ndarray } : A NumPy array where each row represents a document
                            and each column represents a feature:
                            - Letter ratio
                            - Digit ratio
                            - Special character ratio
        """
        try:
            print("Creating character type ratio features...")
            
            features = []

            # Check if in_text is provided, otherwise use the default texts
            text_list = in_text if in_text is not None else self.texts

            if isinstance(text_list, pd.Series):
                text_list = text_list.tolist()  # Convert pandas Series to list if needed
            
            for text in text_list:
                total_chars = len(text)
                
                if total_chars == 0:
                    # If text is empty, append zeros
                    features.append([0, 0, 0])
                    continue
                
                # Count character types
                letter_count = sum(c.isalpha() for c in text)
                digit_count = sum(c.isdigit() for c in text)
                special_count = sum(c in string.punctuation for c in text)
                
                # Compute ratios
                letter_ratio = letter_count / total_chars
                digit_ratio = digit_count / total_chars
                special_ratio = special_count / total_chars
                
                features.append([letter_ratio, digit_ratio, special_ratio])
            
            print(f"Extracted character type ratios for {len(features)} documents.")
            return np.array(features)
        
        except Exception as e:
            print(f"Error: {e}")
            raise


    def create_word_shape_features(self, in_text: str = None) -> np.ndarray:
        """
        Create character-level features based on word shapes.
        
        Returns:
        --------
            { np.ndarray } : A NumPy array where each row represents a document
                            and each column represents a feature:
                            - Capitalized word ratio
                            - All-uppercase word ratio
                            - All-lowercase word ratio
                            - Mixed-case word ratio
                            - Numeric word ratio
                            - Special character word ratio
        """
        try:
            print("Creating word shape features...")
            
            features = []

            # Check if in_text is provided, otherwise use the default texts
            text_list = in_text if in_text is not None else self.texts

            if isinstance(text_list, pd.Series):
                text_list = text_list.tolist()  # Convert pandas Series to list if needed

            for text in text_list:
                words = text.split()
                total_words = len(words)
                
                if total_words == 0:
                    # If text is empty, append zeros
                    features.append([0, 0, 0, 0, 0, 0])
                    continue
                
                # Count word shapes
                capitalized_count = sum(word.istitle() for word in words)  # Starts with uppercase
                all_upper_count = sum(word.isupper() for word in words)   # All uppercase
                all_lower_count = sum(word.islower() for word in words)   # All lowercase
                mixed_case_count = sum(bool(re.search(r'[a-z]', word) and re.search(r'[A-Z]', word)) for word in words)
                numeric_count = sum(word.isdigit() for word in words)     # Fully numeric words
                special_count = sum(bool(re.search(r'\W', word)) for word in words)  # Contains special characters
                
                # Compute ratios
                capitalized_ratio = capitalized_count / total_words
                all_upper_ratio = all_upper_count / total_words
                all_lower_ratio = all_lower_count / total_words
                mixed_case_ratio = mixed_case_count / total_words
                numeric_ratio = numeric_count / total_words
                special_ratio = special_count / total_words
                
                features.append([
                    capitalized_ratio,
                    all_upper_ratio,
                    all_lower_ratio,
                    mixed_case_ratio,
                    numeric_ratio,
                    special_ratio
                ])
            
            print(f"Extracted word shape features for {len(features)} documents.")
            return np.array(features)
        
        except Exception as e:
            print(f"Error: {e}")
            raise





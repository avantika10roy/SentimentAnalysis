# Dependencies
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Feature Engineering
class TextFeatureEngineering:
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
        
        
    def create_binary_bow(self) -> tuple:
        """
        Create binary bag-of-words features (presence/absence)
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted CountVectorizer
                                          - Binary document-term matrix
        """
        try:
            print("Creating binary bag-of-words features...")
            vectorizer = CountVectorizer(binary       = True,
                                         max_features = self.max_features,
                                         ngram_range  = self.ngram_range)
            
            features   = vectorizer.fit_transform(self.texts)
            print(f"Created {features.shape[1]} binary features")
            
            return vectorizer, features
            
        except Exception as e:
            raise
            
            
    def create_count_bow(self) -> tuple:
        """
        Create count-based bag-of-words features
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted CountVectorizer
                                          - Count document-term matrix
        """
        try:
            print("Creating count-based bag-of-words features...")
            vectorizer = CountVectorizer(max_features = self.max_features,
                                         ngram_range  = self.ngram_range)
            
            features   = vectorizer.fit_transform(self.texts)
            print(f"Created {features.shape[1]} count-based features")
            
            return vectorizer, features
            
        except Exception as e:
            raise
            
            
    def create_frequency_bow(self) -> tuple:
        """
        Create frequency-based bag-of-words features (term frequency)
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted TfidfVectorizer
                                          - Term frequency document-term matrix
        """
        try:
            print("Creating frequency-based bag-of-words features...")
            
            vectorizer = TfidfVectorizer(use_idf      = False,
                                         max_features = self.max_features,
                                         ngram_range  = self.ngram_range)
            
            features   = vectorizer.fit_transform(self.texts)
            print(f"Created {features.shape[1]} frequency-based features")
            
            return vectorizer, features
            
        except Exception as e:
            raise
            
            
    def create_tfidf(self) -> tuple:
        """
        Create TF-IDF features
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted TfidfVectorizer
                                          - TF-IDF document-term matrix
        """
        try:
            print("Creating TF-IDF features...")
            vectorizer = TfidfVectorizer(max_features = self.max_features,
                                         ngram_range  = self.ngram_range)
            
            features   = vectorizer.fit_transform(self.texts)
            print(f"Created {features.shape[1]} TF-IDF features")
            
            return vectorizer, features
            
        except Exception as e:
            raise
            
            
    def create_standardized_tfidf(self) -> tuple:
        """
        Create Standardized TF-IDF features
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted TfidfVectorizer
                                          - Standardized TF-IDF document-term matrix
        """
        try:
            print("Creating Standardized TF-IDF features...")
            vectorizer          = TfidfVectorizer(max_features = self.max_features, 
                                                  ngram_range  = self.ngram_range)
            
            tfidf_matrix        = vectorizer.fit_transform(self.texts)
            
            scaler              = StandardScaler(with_mean = False)
            
            standardized_matrix = scaler.fit_transform(tfidf_matrix)
            
            print(f"Created {standardized_matrix.shape[1]} standardized TF-IDF features")
            return vectorizer, standardized_matrix
            
        except Exception as e:
            raise
            
            
    def _create_bm25_variant(self, variant: str, k1: float = 1.5, b: float = 0.75, delta: float = 1.0) -> tuple:
        """
        Unified private method to create BM25 variant features.

        Arguments:
        ----------
            variant      : Specify the BM25 variant ("BM25", "BM25F", "BM25L", "BM25+", "BM25T")
            k1           : Term frequency saturation parameter (default: 1.5)
            b            : Length normalization parameter (default: 0.75)
            delta        : Free parameter for certain variants (default: 1.0)

        Returns:
        --------
            { tuple }    : Tuple containing:
                           - Custom transformer for the specified BM25 variant
                           - BM25 variant document-term matrix
        """
        try:
            print(f"Creating {variant} features...")

            class BM25VariantTransformer(BaseEstimator, TransformerMixin):
                def __init__(self, k1=1.5, b=0.75, delta=1.0, variant="BM25", max_features=None):
                    self.k1               = k1
                    self.b                = b
                    self.delta            = delta
                    self.variant          = variant
                    self.max_features     = max_features
                    self.count_vectorizer = CountVectorizer(max_features = self.max_features)

                def fit(self, texts):
                    # Calculate IDF and average document length
                    X                   = self.count_vectorizer.fit_transform(texts)
                    self.avg_doc_length = X.sum(axis=1).mean()
                    n_docs              = len(texts)
                    df                  = np.bincount(X.indices, minlength=X.shape[1])
                    self.idf            = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                    return self

                def transform(self, texts):
                    X           = self.count_vectorizer.transform(texts)
                    doc_lengths = X.sum(axis=1).A1
                    rows, cols  = X.nonzero()
                    data        = list()

                    for i, j in zip(rows, cols):
                        tf = X[i, j]

                        if (self.variant == "BM25"):
                            numerator   = tf * (self.k1 + 1)
                            denominator = tf + self.k1 * (1 - self.b + self.b * doc_lengths[i] / self.avg_doc_length)
                            score       = self.idf[j] * numerator / denominator
                        
                        elif (self.variant == "BM25F"):
                            score = self.idf[j] * (tf / (self.k1 + tf))

                        elif (self.variant == "BM25L"):
                            numerator   = tf + self.delta
                            denominator = tf + self.delta + self.k1 * (1 - self.b + self.b * doc_lengths[i] / self.avg_doc_length)
                            score       = self.idf[j] * numerator / denominator
                        
                        elif (self.variant == "BM25+"):
                            numerator   = tf + self.delta
                            denominator = tf + self.k1
                            score       = self.idf[j] * numerator / denominator
                        
                        elif (self.variant == "BM25T"):
                            score = self.idf[j] * (tf * np.log(1 + tf))
                        
                        else:
                            raise ValueError(f"Unknown variant: {self.variant}")
                        
                        data.append(score)

                    return csr_matrix((data, (rows, cols)), shape=X.shape)

            transformer = BM25VariantTransformer(k1           = k1, 
                                                 b            = b, 
                                                 delta        = delta,
                                                 variant      = variant, 
                                                 max_features = self.max_features)

            features    = transformer.fit_transform(self.texts)
            print(f"Created {features.shape[1]} {variant} features")
            return transformer, features

        except Exception as e:
            raise

    def create_bm25(self, k1: float = 1.5, b: float = 0.75) -> tuple:
        """
        Create BM25 features
        """
        return self._create_bm25_variant(variant = "BM25", 
                                         k1      = k1, 
                                         b       = b)


    def create_bm25f(self, k1: float = 1.5) -> tuple:
        """
        Create BM25F features
        """
        return self._create_bm25_variant(variant = "BM25F", 
                                         k1      = k1)


    def create_bm25l(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0) -> tuple:
        """
        Create BM25L features
        """
        return self._create_bm25_variant(variant = "BM25L", 
                                         k1      = k1, 
                                         b       = b, 
                                         delta   = delta)


    def create_bm25_plus(self, k1: float = 1.5, delta: float = 1.0) -> tuple:
        """
        Create BM25+ features
        """
        return self._create_bm25_variant(variant = "BM25+", 
                                         k1      = k1, 
                                         delta   = delta)


    def create_bm25t(self, k1: float = 1.5) -> tuple:
        """
        Create BM25T features
        """
        return self._create_bm25_variant(variant = "BM25T", 
                                         k1      = k1)


    def create_skipgrams(self, k: int = 2) -> tuple:
        """
        Create skipgram features
        
        Arguments:
        ----------
            k { int } : Skip distance
            
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted CountVectorizer for skipgrams
                                          - Skipgram document-term matrix
        """
        try:
            print("Creating skipgram features...")
            
            def generate_skipgrams(text: str) -> str:
                words     = text.split()
                skipgrams = list()
                
                for i in range(len(words) - k - 1):
                    skipgram = f"{words[i]}_{words[i + k + 1]}"
                    skipgrams.append(skipgram)
                    
                return ' '.join(skipgrams)
            
            processed_texts = [generate_skipgrams(text) for text in self.texts]
            
            vectorizer      = CountVectorizer(max_features=self.max_features)
            features        = vectorizer.fit_transform(processed_texts)
            
            print(f"Created {features.shape[1]} skipgram features")
            return vectorizer, features
            
        except Exception as e:
            raise
            
            
    def create_positional_ngrams(self) -> tuple:
        """
        Create positional n-gram features
        
        Returns:
        --------
            { tuple } : Tuple containing: - Fitted CountVectorizer for positional n-grams
                                          - Positional n-gram document-term matrix
        """
        try:
            print("Creating positional n-gram features...")
            
            def generate_positional_ngrams(text: str) -> str:
                words      = text.split()
                pos_ngrams = list()
                
                for i in range(len(words)):
                    for n in range(self.ngram_range[0], min(self.ngram_range[1] + 1, len(words) - i + 1)):
                        ngram     = '_'.join(words[i:i+n])
                        pos_ngram = f"pos{i}_{ngram}"
                        pos_ngrams.append(pos_ngram)
                        
                return ' '.join(pos_ngrams)
            
            processed_texts = [generate_positional_ngrams(text) for text in self.texts]
            
            vectorizer      = CountVectorizer(max_features = self.max_features)
            
            features        = vectorizer.fit_transform(processed_texts)
            
            print(f"Created {features.shape[1]} positional n-gram features")
            return vectorizer, features
            
        except Exception as e:
            raise
            
            
    def create_all_features(self) -> dict:
        """
        Create all available feature types
        
        Returns:
        --------
            { dict } : Dictionary mapping feature names to their vectorizer and feature matrix
        """
        try:
            print("Creating all feature types...")
            features                      = dict()
            
            # Create all feature types
            features['binary_bow']        = self.create_binary_bow()
            features['count_bow']         = self.create_count_bow()
            features['frequency_bow']     = self.create_frequency_bow()
            features['tfidf']             = self.create_tfidf()
            features['bm25']              = self.create_bm25()
            features['skipgrams']         = self.create_skipgrams()
            features['positional_ngrams'] = self.create_positional_ngrams()
            
            print("Created all feature types successfully")
            return features
            
        except Exception as e:
            raise
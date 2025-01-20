# Dependencies
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureEngineering_document:
    """
    A class for implementing various advanced text feature engineering techniques
    
    Attributes:
    -----------
        texts        { list }  : List of preprocessed text documents

        max_features  { int }  : Maximum number of features to create

        ngram_range  { tuple } : Range of n-grams to consider
    """
    
    def __init__(self, texts:list, max_features:int = None, ngram_range:tuple = (1, 3)) -> None:
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
        
    def create_lda(self, n_topics:int = 10) -> tuple:
        """
        Create Latent Dirichlet Allocation (LDA) features
        
        Arguments:
        ----------
            n_topics { int } : Number of topics
        
        Returns:
        --------
            { tuple } : Tuple containing:
                        - Fitted LDA model
                        - Document-topic matrix
        """
        try:
            print("Creating LDA features...")
            vectorizer = CountVectorizer(max_features = self.max_features, 
                                         ngram_range  = self.ngram_range)

            X          = vectorizer.fit_transform(self.texts)

            lda_model  = LatentDirichletAllocation(n_components = n_topics, 
                                                   random_state = 42)
            lda_topics = lda_model.fit_transform(X)
            
            print(f"Created {n_topics} LDA topics")
            return lda_model, lda_topics
        
        except Exception as e:
            raise

    def create_lsi(self, n_topics:int = 10) -> tuple:
        """
        Create Latent Semantic Indexing (LSI) features
        
        Arguments:
        ----------
            n_topics { int } : Number of topics
        
        Returns:
        --------
            { tuple } : Tuple containing:
                        - Fitted TruncatedSVD model (LSI)
                        - Document-topic matrix
        """
        try:
            print("Creating LSI features...")
            vectorizer = TfidfVectorizer(max_features = self.max_features, 
                                         ngram_range  = self.ngram_range)

            X          = vectorizer.fit_transform(self.texts)

            lsi_model  = TruncatedSVD(n_components = n_topics, 
                                      random_state = 42)

            lsi_topics = lsi_model.fit_transform(X)
            
            print(f"Created {n_topics} LSI topics")
            return lsi_model, lsi_topics
        
        except Exception as e:
            raise

    def create_document_embeddings(self) -> tuple:
        """
        Create document embeddings using Word2Vec
        
        Returns:
        --------
            { tuple } : Tuple containing:
                        - Word2Vec model
                        - Document embeddings (average of word vectors)
        """
        try:
            print("Creating document embeddings...")
            sentences = [text.split() for text in self.texts]
            model     = Word2Vec(sentences   = sentences, 
                                 vector_size = 100, 
                                 window      = 5, 
                                 min_count   = 1, 
                                 workers     = 4)

            embeddings = np.array([np.mean([model.wv[word] for word in sentence if word in model.wv] or [np.zeros(100)], axis=0)
                                  for sentence in sentences])
            
            print("Created document embeddings")
            return model, embeddings
        
        except Exception as e:
            raise

    def create_document_similarity_matrix(self) -> np.ndarray:
        """
        Create a document similarity matrix based on cosine similarity
        
        Returns:
        --------
            { np.ndarray } : Cosine similarity matrix of documents
        """
        try:
            print("Creating document similarity matrix...")
            vectorizer        = TfidfVectorizer(max_features = self.max_features, 
                                                ngram_range  = self.ngram_range)

            X                 = vectorizer.fit_transform(self.texts)
            
            similarity_matrix = cosine_similarity(X)
            
            print("Created document similarity matrix")
            return similarity_matrix
        
        except Exception as e:
            raise

    def create_hierarchical_document_features(self) -> tuple:
        """
        Create hierarchical document features by applying Agglomerative Clustering
        
        Returns:
        --------
            { tuple } : Tuple containing:
                        - Fitted AgglomerativeClustering model
                        - Document clusters
        """
        try
            print("Creating hierarchical document features...")
            vectorizer       = TfidfVectorizer(max_features = self.max_features, 
                                               ngram_range  = self.ngram_range)

            X                = vectorizer.fit_transform(self.texts)

            clustering_model = AgglomerativeClustering(n_clusters = 5)

            clusters         = clustering_model.fit_predict(X.toarray())
            
            print("Created hierarchical document features")
            return clustering_model, clusters
        
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
            features                          = dict()

            # Create all feature types
            features['lda']                   = self.create_lda()
            features['lsi']                   = self.create_lsi()
            features['document_embeddings']   = self.create_document_embeddings()
            features['document_similarity']   = self.create_document_similarity_matrix()
            features['hierarchical_features'] = self.create_hierarchical_document_features()
            
            print("Created all feature types successfully")
            return features
        
        except Exception as e:
            raise

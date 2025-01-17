# DEPENDENCIES

from gensim.models import Word2Vec
import numpy as np

class Semantic_Feature_Engineering:
    
    """
    A class for implementing various semantic feature engineering techniques.
    
    Attributes:
    -----------
        texts        { list }  : List of preprocessed text documents.
        
        max_features  { int }  : Maximum number of features to create.
    """
    
    def __init__(self, texts: list, max_features: int = None) -> None:
        
        """
        Initialize Semantic_Feature_Engineering with texts and parameters.
        
        Arguments:
        ----------
            texts        : List of preprocessed text documents.
            
            max_features : Maximum number of features (None for no limit).
            
        Raises:
        -------
            ValueError   : If texts is empty or parameters are invalid.
        """
        
        if not texts:
            raise ValueError("Input texts cannot be empty")
            
        self.texts        = texts
        self.max_features = max_features
    
    def word2vec_cbow(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4) -> tuple:
        
        """
        Generate semantic features using Word2Vec (CBOW) and return the feature matrix and vectorizer.
        
        Arguments:
        ----------
            vector_size : Dimensionality of word embeddings (default: 100).
            
            window      : Context window size (default: 5).
            
            min_count   : Ignores words with frequency lower than this (default: 1).
            
            workers     : Number of worker threads to train the model (default: 4).
        
        Returns:
        --------
            tuple:
                - np.ndarray : Document-level feature matrix (each document represented as the average of its word vectors).
                - Word2Vec   : The trained Word2Vec model (vectorizer).
        """
    
        tokenized_texts          = [doc.split() for doc in self.texts]
        
        w2v_model                = Word2Vec(sentences   = tokenized_texts, 
                                            vector_size = vector_size, 
                                            window      = window, 
                                            min_count   = min_count, 
                                            workers     = workers,
                                            sg          = 0
                                            )
        
        features                 = []
        
        for tokens in tokenized_texts:
            
            vectors              = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
            
            if vectors:
                document_vector  = np.mean(vectors, axis=0)
            
            else:
                document_vector  = np.zeros(vector_size)
            
            features.append(document_vector)
        
        feature_matrix           = np.array(features)
        
        if self.max_features is not None and self.max_features < vector_size:
            feature_matrix       = feature_matrix[:, :self.max_features]
        
        return feature_matrix, w2v_model
    
    def glove(self, glove_path: str, embedding_dim: int = 100) -> tuple:
        
        """
        Generate semantic features using GloVe and return the feature matrix and embedding dictionary.
        
        Arguments:
        ----------
            glove_path        : Path to the GloVe embeddings file.
            
            embedding_dim     : Dimensionality of GloVe embeddings (default: 100).
        
        Returns:
        --------
            tuple:
                - np.ndarray  : Document-level feature matrix (each document represented as the average of its word vectors).
                - dict        : The GloVe embedding dictionary.
        """
    
        glove_embeddings                = {}
        
        with open(glove_path, 'r', encoding = 'utf-8') as f:
            
            for line in f:
                values                  = line.split()
                word                    = values[0]
                vector                  = np.asarray(values[1:], dtype='float32')
                glove_embeddings[word]  = vector
        
        tokenized_texts                 = [doc.split() for doc in self.texts]
        
        features                        = []
        
        for tokens in tokenized_texts:
    
            vectors                     = [glove_embeddings[word] for word in tokens if word in glove_embeddings]
            
            if vectors:
                document_vector         = np.mean(vectors, axis=0)
            
            else:
                document_vector         = np.zeros(embedding_dim)
            
            features.append(document_vector)
        
        feature_matrix                  = np.array(features)
        
        if self.max_features is not None and self.max_features < embedding_dim:
            feature_matrix              = feature_matrix[:, :self.max_features]
        
        return feature_matrix, glove_embeddings
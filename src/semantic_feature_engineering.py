## ----- DONE BY PRIYAM PAL -----


# DEPENDENCIES

import torch
import numpy as np

import tensorflow_hub as hub

from gensim.models import Word2Vec
from gensim.models import FastText
from nltk.corpus import wordnet as wn


# ----- SEMANTIC FEATURE ENGINEERING -----
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
    
    # ----- WORD2VEC MODEL -----
    
    def word2vec_cbow(self, vector_size: int = None, window: int = 5, min_count: int = 1, workers: int = 4) -> tuple:
        """
        Generate semantic features using Word2Vec (CBOW) and return the vectorizer and feature matrix.
    
        Arguments:
        ----------
            vector_size      : Dimensionality of word embeddings (default: 100).
            window           : Context window size (default: 5).
            min_count        : Ignores words with frequency lower than this (default: 1).
            workers          : Number of worker threads to train the model (default: 4).
    
        Returns:
        --------
            tuple:
                - Word2Vec   : The trained Word2Vec model (vectorizer).
                - np.ndarray : Document-level feature matrix (each document represented as the average of its word vectors).
        """
        try:
            print("Creating Word2Vec (CBOW) features...")
            
            if vector_size is None:
                vector_size = self.max_features
            
            tokenized_texts         = [doc.split() for doc in self.texts]
            
            max_features            = self.max_features

            w2v_model               = Word2Vec(sentences   = tokenized_texts,
                                               vector_size = vector_size,
                                               window      = window,
                                               min_count   = min_count,
                                               workers     = workers,
                                               sg          = 0
                                               )
        
            features                = []
            
            for tokens in tokenized_texts:
                vectors             = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
            
                if vectors:
                    document_vector = np.mean(vectors, axis=0)
                else:
                    document_vector = np.zeros(vector_size)
            
                features.append(document_vector)
        
            feature_matrix          = np.array(features, dtype=np.float32)

            if self.max_features is not None and self.max_features < vector_size:
                feature_matrix      = feature_matrix[:, :self.max_features]
        
            print(f"Created Word2Vec (CBOW) features with shape: {feature_matrix.shape}")
        
            return w2v_model, feature_matrix

        except Exception as e:
            raise Exception(f"Error in creating Word2Vec (CBOW) features: {str(e)}")
    
      # ----- GLOVE EMBEDDING -----
    
    def glove(self, glove_path: str, embedding_dim: int = 100, desired_features: int = None) -> tuple:
        """
        Generate semantic features using GloVe and return the feature matrix and embedding dictionary.
    
        Arguments:
        ----------
            glove_path        : Path to the GloVe embeddings file.
            embedding_dim     : Dimensionality of GloVe embeddings (default: 100).
            desired_features  : Number of features to extract (default: 10000).
    
        Returns:
        --------
            tuple:
                - np.ndarray  : Document-level feature matrix (each document represented as the average of its word vectors).
                - dict        : The GloVe embedding dictionary.
        """

        try:
            print("Creating GloVe features...")
            
            if desired_features is None:
                desired_features = self.max_features
        
            glove_embeddings               = {}
            
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values                 = line.split()
                    word                   = values[0]
                    vector                 = np.asarray(values[1:], dtype='float32')
                    glove_embeddings[word] = vector

            tokenized_texts                = [doc.split() for doc in self.texts]
        
            features                       = []
        
            for tokens in tokenized_texts:
                vectors                    = [glove_embeddings[word] for word in tokens if word in glove_embeddings]
            
                if vectors:
                 document_vector           = np.mean(vectors, axis=0)
                
                else:
                   document_vector         = np.zeros(embedding_dim)
            
            features.append(document_vector)
        
            feature_matrix                 = np.array(features)
        
            if desired_features < embedding_dim:
                feature_matrix             = feature_matrix[:, :desired_features]
        
            print(f"GloVe features created with shape: {feature_matrix.shape}")
        
            return glove_embeddings, feature_matrix

        except Exception as e:
            raise Exception(f"Error in creating GloVe features: {str(e)}")
    
    # ----- FAST-TEXT VECTORIZER ------
    
    def fasttext(self, vector_size: int = None, window: int = 5, min_count: int = 1, workers: int = 4, precision: type = np.float32) -> tuple:
        """
        Generate semantic features using the FastText model (skip-gram) and return the vectorizer and feature matrix.
    
        Arguments:
        ----------
            vector_size       : Dimensionality of word embeddings (default: 10000).
            window            : Context window size (default: 5).
            min_count         : Ignores words with frequency lower than this (default: 1).
            workers           : Number of worker threads to train the model (default: 4).
            precision         : Data type for the feature matrix (default: np.float32).
    
        Returns:
        --------
            tuple:
                - FastText    : The trained FastText model (vectorizer).
                - np.ndarray  : Document-level feature matrix (each document represented as the average of its word vectors).
        """
        try:
            print("Creating FastText (skip-gram) features...")
            
            if vector_size is None:
                vector_size = self.max_features
        
            tokenized_texts         = [doc.split() for doc in self.texts]
        
            max_features            = self.max_features

            fasttext_model          = FastText(sentences   = tokenized_texts, 
                                               vector_size = vector_size, 
                                               window      = window, 
                                               min_count   = min_count, 
                                               workers     = workers, 
                                               sg          = 1
                                               )

            features                = []
        
            for tokens in tokenized_texts:
                vectors             = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
            
                if vectors:
                    document_vector = np.mean(vectors, axis=0)
                else:
                    document_vector = np.zeros(vector_size)
            
                features.append(document_vector)
        
            feature_matrix           = np.array(features, precision)

            if self.max_features is not None and self.max_features < vector_size:
                feature_matrix       = feature_matrix[:, :self.max_features]
        
            print(f"Created FastText (skip-gram) features with shape: {feature_matrix.shape}")
        
            return fasttext_model, feature_matrix

        except Exception as e:
            raise Exception(f"Error in creating FastText (skip-gram) features: {str(e)}")
    
    
    # ----- CONTEXTUAL EMBEDDING -----
    class Contextual_Embedding:
        """
        
        Class to generate Contextual Embeddings using various models using ELMo, BERT, GPT.
        
        """
        
        def __init__(self, texts: list):
            
            """
            Initialize Semantic_Feature_Engineering with texts and parameters.
        
            Arguments:
            ----------
                texts        : List of preprocessed text documents.
            
            Raises:
            -------
                ValueError   : If texts is empty or parameters are invalid.
            """
            if not texts:
                raise ValueError("Input texts cannot be empty")
            
            self.texts        = texts
            self.max_features = None
        
        # ----- ELMO EMBEDDING -----
        
        def elmo(self, model_url: str, batch_size: int = 32) -> np.ndarray:
            
            """
            Generate contextual embeddings using ELMo model and return the feature matrix.
            
            Arguments:
            ----------
                options_file   : Path to the ELMo options file.
                weight_file    : Path to the ELMo pre-trained weights file.
                batch_size     : Batch size for processing text (default: 32).
        
            Returns:
            --------
                - np.ndarray   : Document-level feature matrix (average ELMo embeddings for each document).
                - elmo         : The trained ELMo model (vectorizer).
            """
    
            # Load the pre-trained ELMo model from TensorFlow Hub
            elmo_model = hub.load(model_url)

            
            document_embeddings  = []
            
            for text in self.texts:
                tokens           = text.split()
                
                if len(tokens) == 0:
                    document_embeddings.append(np.zeros(1024)) 
                    continue
                
                embeddings       = elmo_model.signatures["default"](input={"tokens": tokens})["output"]
                
                document_vector  = np.mean(embeddings, axis=0)
                
                document_embeddings.append(document_vector)
            
            feature_matrix       = np.array(document_embeddings)

            if self.max_features is not None and self.max_features < 1024:
                feature_matrix  = feature_matrix[:, :self.max_features]
            
            return feature_matrix, elmo_model
    
    # ------ WORDNET FEATURES -----
        
    def wordnet(self) -> tuple:
    
        """
        Generate semantic features using WordNet, including synonyms, hypernyms, hyponyms, 
        and meronyms, and return the feature list and WordNet corpus.
    
        Arguments:
        ----------
            None
    
        Returns:
        --------
         tuple:
             - list                                           : A list of dictionaries where each dictionary 
                                                                contains synonyms, hypernyms, hyponyms, and 
                                                                meronyms for words in the corresponding document.
                                                                
             - nltk.corpus.reader.wordnet.WordNetCorpusReader : The WordNet corpus used for feature extraction.
        """
    
    
        wordnet_features   = []
    
        for doc in self.texts:
            doc_features   = {"synonyms": [], "hypernyms": [], "hyponyms": [], "meronyms": []}
        
            for word in doc.split():
                synsets    = wn.synsets(word)
            
                # SYNOMYMS
                synonyms   = set()
                for synset in synsets:
                    synonyms.update(lemma.name() for lemma in synset.lemmas())
                    
                # HYPERNYMS
                hypernyms  = set()
                for synset in synsets:
                    hypernyms.update(lemma.name() for hyper in synset.hypernyms() for lemma in hyper.lemmas())
            
                # HYPONYMS
                hyponyms   = set()
                for synset in synsets:
                    hyponyms.update(lemma.name() for hypo in synset.hyponyms() for lemma in hypo.lemmas())
            
                # MERONYMS
                meronyms   = set()
                for synset in synsets:
                    meronyms.update(lemma.name() for mero in synset.part_meronyms() for lemma in mero.lemmas())
            
                doc_features["synonyms"].append(list(synonyms))
                doc_features["hypernyms"].append(list(hypernyms))
                doc_features["hyponyms"].append(list(hyponyms))
                doc_features["meronyms"].append(list(meronyms))
        
            wordnet_features.append(doc_features)
    
        return wn, wordnet_features
    
    
# Dependencies
import nltk
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('punkt')

class ClassFeatureEngineering:

    """
    A class for implementing various text feature engineering techniques
    
    Attributes:
    -----------
        texts        { list }  : List of preprocessed text documents
        labels       { list }  : List of class labels corresponding to each text document
    """
    
    def __init__(self, texts: list, labels: list) -> None:
        """
        Initialize ClassFeatureEngineering with texts and labels
        
        Arguments:
        ----------
            texts   : List of preprocessed text documents
            labels  : List of class labels corresponding to each text document
            
        Raises:
        -------
            ValueError   : If texts or labels are empty or of different lengths
        """
        if not texts or not labels:
            raise ValueError("Input texts and labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError("The number of texts and labels must be the same")
        
        self.texts  = texts
        self.labels = labels
        self.classes = np.unique(labels)


    def class_specific_vocabulary(self) -> tuple:
        """
        Generate a vocabulary specific to each class label and return a vectorizer and sparse matrix.
        
        Returns:
        --------
            tuple : (vectorizer, sparse_matrix) 
                - vectorizer : A fitted CountVectorizer instance.
                - sparse_matrix : A sparse matrix with the term frequencies for each class.
        """
        try:
            print("Creating class-specific vocabulary...")
            class_vocabs = defaultdict(set)
            
            # Collect class-specific vocabularies
            for text, label in zip(self.texts, self.labels):
                tokens = word_tokenize(text.lower())  
                class_vocabs[label].update(tokens)
            
            print("Class-specific vocabulary created.")
            
            # Create a unified list of all class-specific vocabularies
            all_tokens = [' '.join(list(vocab)) for vocab in class_vocabs.values()]
            
            # Initialize and fit the CountVectorizer
            vectorizer = CountVectorizer()
            sparse_matrix = vectorizer.fit_transform(all_tokens)
            
            return vectorizer, sparse_matrix
            
        except Exception as e:
            raise e
        

    def label_aware_embeddings(self, embedding_dim=100) -> tuple:
        """
        Generate label-aware embeddings and return vectorizer and sparse matrix.
        
        Parameters:
        -----------
            embedding_dim : int
                Dimensionality of the embedding vectors.
        
        Returns:
        --------
            tuple : (vectorizer, sparse_matrix)
                - vectorizer : Fitted CountVectorizer for label text.
                - sparse_matrix : Sparse matrix representation of label embeddings.
        """
        try:
            print("Generating label-aware embeddings...")
            
            # Tokenize the texts
            tokenized_texts = [word_tokenize(text.lower()) for text in self.texts]
            
            # Train Word2Vec embeddings
            w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, window=5, min_count=1, workers=4)
            
            # Prepare label-specific texts
            label_texts = {label: ' '.join([word for text, lbl in zip(self.texts, self.labels) if lbl == label for word in word_tokenize(text.lower())]) for label in set(self.labels)}
            
            # Use CountVectorizer to create sparse matrix
            vectorizer = CountVectorizer()
            sparse_matrix = vectorizer.fit_transform(label_texts.values())
            
            print("Label-aware embeddings generated.")
            return vectorizer, sparse_matrix
        
        except Exception as e:
            raise e


    """

    def hierarchical_class_features(self, hierarchy=None) -> tuple:
    
    Generate hierarchical class features and return vectorizer and sparse matrix.
        
    Parameters:
        -----------
            hierarchy : dict
                A dictionary where keys are parent labels and values are lists of child labels.
        
        Returns:
        --------
            tuple : (vectorizer, sparse_matrix)
                - vectorizer : Fitted CountVectorizer for hierarchical features.
                - sparse_matrix : Sparse matrix representation of hierarchical features.
    
        try:
            print("Generating hierarchical class features...")
            
            # Flatten hierarchy into parent-child paths
            parent_child_pairs = []
            for parent, children in hierarchy.items():
                for child in children:
                    parent_child_pairs.append((parent, child))
            
            # Create a mapping of labels to their hierarchical paths
            label_to_hierarchy = defaultdict(list)
            for parent, child in parent_child_pairs:
                label_to_hierarchy[child].append(parent)
                label_to_hierarchy[child].extend(label_to_hierarchy[parent])  # Add parent's hierarchy recursively
            
            # Flatten paths into text representations
            hierarchical_texts = {label: ' '.join(path) for label, path in label_to_hierarchy.items()}
            
            # Use CountVectorizer to create sparse matrix
            vectorizer = CountVectorizer()
            sparse_matrix = vectorizer.fit_transform(hierarchical_texts.values())
            
            print("Hierarchical class features generated.")
            return vectorizer, sparse_matrix
        
        except Exception as e:
            raise e
        
    """
    def multi_label_features(self) -> tuple:
        
        """
            Generate multi-label features and return vectorizer and sparse matrix.
            
            Returns:
            --------
                tuple : (vectorizer, sparse_matrix)
                    - vectorizer : Fitted CountVectorizer for multi-label features.
                    - sparse_matrix : Sparse matrix representation of multi-label features.
        """
        try:
            print("Generating multi-label features...")
                
            # Convert labels to sets if not already
            multi_labels = [set(lbl) if isinstance(lbl, list) else {lbl} for lbl in self.labels]
                
            # Create multi-label strings for each sample
            label_texts = [' '.join(label) for label in multi_labels]
                
            # Use CountVectorizer to create sparse matrix
            vectorizer = CountVectorizer()
            sparse_matrix = vectorizer.fit_transform(label_texts)
                
            print("Multi-label features generated.")
            return vectorizer, sparse_matrix
            
        except Exception as e:
            raise e
        
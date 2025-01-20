## ----- DONE BY PRIYAM PAL -----


# DEPENDENCIES

import numpy as np

from scipy.sparse import hstack
from scipy.sparse import csr_matrix


# ----- TRANFORMING THE VECTORIZERS -----

def vector_transform(texts, model):
    """
    Transform a list of texts into sentence embeddings using a Word2Vec or FastText model.

    Args:
        texts: List of sentences (strings).
        model: Pre-trained Word2Vec or FastText model.

    Returns:
        NumPy array of sentence embeddings.
    """
   
    transformed            = []
    
    for text in texts:
        words              = text.split()
        word_vectors       = [model.wv[word] for word in words if word in model.wv]
        
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        
        else:
            sentence_vector = np.zeros(model.vector_size)
        
        transformed.append(sentence_vector)
    
    return np.array(transformed)
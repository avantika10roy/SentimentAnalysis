## ----- DONE BY PRIYAM PAL -----


# DEPENDENCIES

import torch
import numpy as np


def vector_transform(texts, model, tokenizer = None, model_type = None, max_seq_length = 128):
    
    """
    Transform a list of texts into sentence embeddings using Word2Vec, FastText, or Transformer-based models.

    Arguments:
    ----------
    
        texts           : List of sentences (strings).
        model           : Pre-trained Word2Vec/FastText model or Transformer model (e.g., BERT).
        tokenizer       : Tokenizer for Transformer models (required if model_type is 'transformer').
        model_type      : Type of the model ('word2vec' or 'transformer').
        max_seq_length  : Maximum sequence length for Transformer models (default: 128).

    Returns:
    ----------
    
        NumPy array of sentence embeddings.
    """
    
    transformed = []

    if model_type == "word2vec" or model_type == "fasttext":
        
        for text in texts:
            words                = text.split()
            word_vectors         = [model.wv[word] for word in words if word in model.wv]

            if word_vectors:
                sentence_vector  = np.mean(word_vectors, axis=0)
            else:
                sentence_vector  = np.zeros(model.vector_size)

            transformed.append(sentence_vector)

    elif model_type == "distilbert" or model_type == "bert":
        
        if tokenizer is None:
            raise ValueError("Tokenizer is required for Transformer models.")

        model.eval()
        
        with torch.no_grad():
            for text in texts:
                encoded        = tokenizer(text, 
                                           max_length     = max_seq_length,
                                           padding        = "max_length",
                                           truncation     = True,
                                           return_tensors = "pt",)
                input_ids      = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                outputs        = model(input_ids      = input_ids, 
                                       attention_mask = attention_mask)

                cls_embedding  = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
                
                transformed.append(cls_embedding)

    else:
        raise ValueError("Unsupported model_type. Use 'word2vec' or 'transformer'.")

    return np.array(transformed, dtype=np.float32)
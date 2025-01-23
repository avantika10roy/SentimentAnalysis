# -- Done By Manu Bhaskar --

# -- Dependencies -- 
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

class TextVectorizer:
    def __init__(
        self, feature_names=None, weight_factor=2.0, vector_size=100, use_pretrained=False, pretrained_path=None
    ):
        """
        Initialize the TextVectorizer class.

        :param feature_names: List of feature names (tokens) to be weighted more.
        :param weight_factor: Weight multiplier for feature names.
        :param vector_size: Dimensionality of word vectors (only for training a new model).
        :param use_pretrained: Whether to use a pre-trained Word2Vec model.
        :param pretrained_path: Path to the pre-trained Word2Vec model file (binary format).
        """
        self.feature_names = set(feature_names) if feature_names else set()
        self.weight_factor = weight_factor
        self.vector_size = vector_size
        self.use_pretrained = use_pretrained
        self.pretrained_path = pretrained_path
        self.model = None
        self.bigram_phraser = None
        self.trigram_phraser = None

    def load_pretrained_model(self):
        """
        Load a pre-trained Word2Vec model.
        """
        if not self.pretrained_path:
            raise ValueError("Path to the pre-trained model must be provided.")
        print("Loading pre-trained Word2Vec model...")
        self.model = KeyedVectors.load(self.pretrained_path)
        self.vector_size = self.model.vector_size
        print("Pre-trained Word2Vec model loaded successfully.")

    
    def train(self, corpus):
        """
        Train the CBOW model on the given corpus.

        :param corpus: List of tokenized texts (list of lists of strings).
        """
        sentences = [text.split() for text in corpus]  # Tokenized text
        bigram = Phrases(sentences, min_count=5, threshold=10)
        trigram = Phrases(bigram[sentences], threshold=10)
        
        self.bigram_phraser = Phraser(bigram)
        self.trigram_phraser = Phraser(trigram)
        
        # Transform sentences to include phrases
        processed_corpus = [self.trigram_phraser[self.bigram_phraser[sentence]] for sentence in sentences]

        print("Vectorizer training...")

        self.model = Word2Vec(
            sentences   =processed_corpus,
            vector_size =self.vector_size,
            window      = self.window,
            min_count   = self.min_count,
            sg          = self.n_grams,
            seed        = self.random_state,
            workers     = 1
        )

        
        self.model.train(processed_corpus, total_examples=len(processed_corpus), epochs=self.epochs)
        print("Vectorizer Training Complete")

    def _get_weighted_vector(self, word):
        """
        Get the weighted vector for a given word.

        :param word: Word for which the vector is to be retrieved.
        :return: Weighted vector for the word.
        """
        if word in self.model.key_to_index and word in self.feature_names:
            vector = self.model[word].copy()
            if word in self.feature_names:
                vector *= self.weight_factor
            return vector
        else:
            return np.zeros(self.vector_size)

    def text_to_vector(self, text):
        """
        Convert a text into its vector representation.

        :param text: List of words (tokens) in the text.
        :return: Vector representation of the text.
        """
        vectors = [self._get_weighted_vector(word) for word in text]
        if vectors:
            return np.mean(vectors, axis=0)  # Average the vectors
        else:
            return np.zeros(self.vector_size)

    def transform(self, texts):
        """
        Transform a list of texts into their vector representations.

        :param texts: List of tokenized texts (list of lists of strings).
        :return: List of vector representations of the texts.
        """
        sentences = [text.split() for text in texts]  # Tokenized text
        bigram = Phrases(sentences, min_count=5, threshold=10)
        trigram = Phrases(bigram[sentences], threshold=10)
        
        self.bigram_phraser = Phraser(bigram)
        self.trigram_phraser = Phraser(trigram)
        tokenized_texts = [text.split() for text in texts]
        processed_texts = [self.trigram_phraser[self.bigram_phraser[text]] for text in tokenized_texts]
        return [self.text_to_vector(text) for text in processed_texts]

# Dependencies
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Text Preprocessing
class TextPreprocessor:
    """
    A class for preprocessing text data through cleaning, tokenization, and normalization
    
    Attributes:
    -----------
        lemmatizer : WordNetLemmatizer instance for word lemmatization
        
        stop_words : Set of stopwords to be removed from text
    """ 
    def __init__(self):
        """
        Initialize the TextPreprocessor with required NLTK resources
        
        Raises:
        -------
            LookupError : If required NLTK resources cannot be downloaded
        """
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
        except LookupError as e:
            raise
    
    def clean_text(self, text:str) -> str:
        """
        Clean and normalize input text by removing HTML tags, special characters,
        and applying text normalization techniques
        
        Arguments:
        ----------
            text { str }      : Input text to be cleaned
            
        Raises:
        -------
            ValueError        : If input text is None or empty
            
            TextCleaningError : If any error occurs at any step of text cleaning process
            
        Returns:
        --------
                { str }       : Cleaned and normalized text
        """
        if ((not text) or (not isinstance(text, str))):
            raise ValueError("Input text must be a non-empty string")
            
        try:
            # Remove HTML tags
            text   = re.sub('<[^>]*>', '', text)
            
            # Remove special characters and digits
            text   = re.sub('[^a-zA-Z\s]', '', text)
            
            # Convert to lowercase
            text   = text.lower()
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            
            return ' '.join(tokens)
        
        except Exception as TextCleaningError:
            raise
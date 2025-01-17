import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict

class OptimizedSentimentFeatures:
    '''
    Optimized class for obtaining sentiment features
    '''
    def __init__(self, path="data/emotion_lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"):
        # Load spaCy model with only necessary components
        self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        
        # Preprocess NRC lexicon for faster lookups
        self._load_nrc_lexicon(path)
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Create a batch processor for spaCy
        self.doc_pipe = self.nlp.pipe

    def _load_nrc_lexicon(self, path):
        '''
        Optimized loading of NRC lexicon into a more efficient format
        '''
        nrc_df = pd.read_csv(path, sep='\t', header=None, names=['word', 'emotion', 'association'])
        nrc_filtered = nrc_df[nrc_df['association'] == 1]
        
        # Convert to dictionary for O(1) lookups
        self.emotion_dict = defaultdict(list)
        for _, row in nrc_filtered.groupby('word')['emotion'].apply(list).items():
            self.emotion_dict[_] = row

    def batch_analyze_sentiment(self, texts, batch_size=1000):
        '''
        Batch process multiple texts for better performance
        '''
        results = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
        return results

    def _process_batch(self, texts):
        '''
        Process a batch of texts simultaneously
        '''
        results = []
        
        # Get VADER sentiments for entire batch
        vader_sentiments = [self.vader.polarity_scores(text) for text in texts]
        
        # Process all texts through spaCy pipeline at once
        docs = list(self.doc_pipe(texts))
        
        # Analyze each document
        for doc, vader_sent in zip(docs, vader_sentiments):
            result = {}
            
            # VADER sentiment (already calculated)
            result['vader_sentiment'] = {
                'positive': vader_sent['pos'],
                'negative': vader_sent['neg'],
                'neutral': vader_sent['neu'],
                'compound': vader_sent['compound']
            }
            
            # Emotion counts and aspects in single pass
            emotion_counts = defaultdict(int)
            aspects = []
            
            for token in doc:
                # Emotion analysis
                word = token.text.lower()
                if word in self.emotion_dict:
                    for emotion in self.emotion_dict[word]:
                        emotion_counts[emotion] += 1
                
                # Collect potential aspects (nouns)
                if token.pos_ == "NOUN":
                    aspects.append(token.text)
            
            result['emotion_counts'] = dict(emotion_counts)
            
            # Aspect sentiment (only for nouns now, not chunks)
            aspect_sentiments = {}
            for aspect in aspects:
                sentiment = self.vader.polarity_scores(aspect)
                aspect_sentiments[aspect] = sentiment['compound']
            
            result['aspect_based_sentiment'] = aspect_sentiments
            
            # Polarity patterns (reuse VADER results)
            result['polarity_patterns'] = self._get_polarity(vader_sent)
            
            results.append(result)
        
        return results

    def _get_polarity(self, vader_sent):
        '''
        Determine polarity based on compound score
        '''
        polarity = {
            'positive': vader_sent['pos'],
            'negative': vader_sent['neg'],
            'neutral': vader_sent['neu']
        }
        
        compound_score = vader_sent['compound']
        if compound_score > 0.05:
            polarity['overall_sentiment'] = 'positive'
        elif compound_score < -0.05:
            polarity['overall_sentiment'] = 'negative'
        else:
            polarity['overall_sentiment'] = 'neutral'
            
        return polarity

if __name__ == "__main__":
    # Example usage with batch processing
    texts = [
        "I absolutely love this product! It's amazing.",
        "I am so happy and excited to be here, but a bit scared at the same time.",
        # Add more texts here
    ]
    
    analyzer = OptimizedSentimentFeatures()
    results = analyzer.batch_analyze_sentiment(texts)
    print(results)
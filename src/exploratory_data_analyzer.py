# Dependencies
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


# Exploratyory Data Analysis (EDA)
class SentimentEDA:
    """
    A class for comprehensive Exploratory Data Analysis of sentiment-based text data
    
    Attributes:
    -----------
        df            { DataFrame } : DataFrame containing text and sentiment data
        
        text_column      { str }    : Name of column containing cleaned text
        
        sentiment_column { str }    : Name of column containing sentiment labels
    """
    
    def __init__(self, df: pd.DataFrame, text_column:str = 'clean_text', sentiment_column:str = 'sentiment',
                 output_dir:str = None) -> None:
        """
        Initialize the SentimentEDA class
        
        Arguments:
        ----------
            df           { DataFrame } : Input DataFrame
            
            text_column      { str }   : Name of text column
            
            sentiment_column { str }   : Name of sentiment column
            
            output_dir       { str }   : Directory to save plots
            
        Raises:
        -------
            ValueError                 : If required columns are not in DataFrame
        """
        if ((text_column not in df.columns) or (sentiment_column not in df.columns)):
            raise ValueError(f"DataFrame must contain columns: {text_column} and {sentiment_column}")
        
        # Initialize the Attributes
        self.df               = df
        self.text_column      = text_column
        self.sentiment_column = sentiment_column
        self.output_dir       = output_dir
        
        if output_dir:
            Path(output_dir).mkdir(parents  = True, 
                                   exist_ok = True)
            
        
        
    def save_plot(self, plt: plt, plot_name: str) -> None:
        """
        Helper method to save plots if output directory is specified
        
        """
        if self.output_dir:
            plt.savefig(fname       = f"{self.output_dir}/{plot_name}.png", 
                        bbox_inches = 'tight', 
                        dpi         = 300)
            
            # Print the success statement on screen
            print(f"Saved plot: {plot_name}")
    
    
    def text_length_analysis(self) -> None:
        """
        Analyze and visualize text length distributions
        
        - Character length distribution
        
        - Word length distribution
        
        - Sentence length distribution
        
        """
        try:
            # Calculate lengths
            self.df['char_length']     = self.df[self.text_column].str.len()
            self.df['word_length']     = self.df[self.text_column].str.split().str.len()
            self.df['sentence_length'] = self.df[self.text_column].str.split('.').str.len()
            
            # Create subplots
            fig, axes                  = plt.subplots(nrows   = 3, 
                                                      ncols   = 1, 
                                                      figsize = (12, 15))
            
            # Character length
            sns.boxplot(x    = self.sentiment_column, 
                        y    = 'char_length', 
                        data = self.df, 
                        ax   = axes[0])
            
            axes[0].set_title('Character Length Distribution by Sentiment')
            axes[0].set_ylabel('Number of Characters')
            
            # Word length
            sns.boxplot(x    = self.sentiment_column, 
                        y    = 'word_length', 
                        data = self.df, 
                        ax   = axes[1])
            
            axes[1].set_title('Word Length Distribution by Sentiment')
            axes[1].set_ylabel('Number of Words')
            
            # Sentence length
            sns.boxplot(x    = self.sentiment_column, 
                        y    = 'sentence_length', 
                        data = self.df, 
                        ax   = axes[2])
            
            axes[2].set_title('Sentence Length Distribution by Sentiment')
            axes[2].set_ylabel('Number of Sentences')
            
            plt.tight_layout()
            self.save_plot(plt, 'length_distributions')
            plt.close()
            
            # Print summary statistics
            stats = self.df.groupby(self.sentiment_column)[['char_length', 
                                                            'word_length', 
                                                            'sentence_length']].describe()
            
            print("\nLength Statistics by Sentiment:")
            print(f"\n{stats}")
            
        except Exception as e:
            raise
            
            
    def word_frequency_analysis(self, top_n: int = 20) -> None:
        """
        Analyze and visualize word frequencies by sentiment
        
        Arguments:
        ----------
            top_n { int } : Number of top words to display
        """
        try:
            # Split by sentiment
            sentiment_texts  = {sentiment: ' '.join(group[self.text_column]) for sentiment, group \
                                in self.df.groupby(self.sentiment_column)}
            
            # Create word frequency plots for each sentiment
            fig, axes        = plt.subplots(nrows   = len(sentiment_texts), 
                                            ncols   = 1, 
                                            figsize = (12, 5*len(sentiment_texts)))
            
            for idx, (sentiment, text) in enumerate(sentiment_texts.items()):
                words        = text.split()
                word_freq    = Counter(words).most_common(top_n)
                words, freqs = zip(*word_freq)
                
                ax           = axes[idx] if len(sentiment_texts) > 1 else axes
                
                sns.barplot(x  = list(freqs), 
                            y  = list(words), 
                            ax = ax)
                
                ax.set_title(f'Top {top_n} Words for {sentiment} Sentiment')
                ax.set_xlabel('Frequency')
                
            plt.tight_layout()
            self.save_plot(plt, 'word_frequencies')
            plt.close()
            
        except Exception as e:
            raise
            
            
    def generate_wordclouds(self) -> None:
        """
        Generate and display wordclouds for each sentiment category
        
        """
        try:
            # Create wordcloud for each sentiment
            sentiment_texts = {sentiment: ' '.join(group[self.text_column]) for sentiment, group in self.df.groupby(self.sentiment_column)}
            
            fig, axes       = plt.subplots(1, len(sentiment_texts), figsize=(15, 8))
            
            for idx, (sentiment, text) in enumerate(sentiment_texts.items()):
                wordcloud = WordCloud(width            = 800, 
                                      height           = 400,
                                      background_color = 'white',
                                      max_words        = 150).generate(text)
                
                ax        = axes[idx] if len(sentiment_texts) > 1 else axes
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'{sentiment} Sentiment WordCloud')
                
            plt.tight_layout()
            self.save_plot(plt, 'wordclouds')
            plt.close()
            
        except Exception as e:
            raise
            
            
    def sentiment_intensity_analysis(self) -> None:
        """
        Analyze sentiment intensity using TextBlob's polarity scores
        """
        try:
            # Calculate polarity scores
            self.df['polarity'] = self.df[self.text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
            
            # Create distribution plot
            plt.figure(figsize = (10, 6))
            
            sns.kdeplot(data = self.df, 
                        x    = 'polarity', 
                        hue  = self.sentiment_column)
            
            plt.title('Sentiment Polarity Distribution')
            plt.xlabel('Polarity Score')
            plt.ylabel('Density')

            self.save_plot(plt, 'sentiment_intensity')
            plt.close()
            
            # Print summary statistics
            stats = self.df.groupby(self.sentiment_column)['polarity'].describe()
            print("\nPolarity Statistics by Sentiment:")
            print(f"\n{stats}")
            
        except Exception as e:
            raise
        
        
    def pos_distribution_analysis(self) -> None:
        """
        Analyze distribution of Parts of Speech across sentiments
        """
        try:
            nlp = spacy.load('en_core_web_sm')
            
            def get_pos_counts(text: str):
                doc = nlp(text)
                return Counter([token.pos_ for token in doc])
            
            # Get POS counts for sample of texts (for efficiency)
            sample_size                   = min(1000, len(self.df))
            sample_df                     = self.df.sample(sample_size, random_state=42)
            
            pos_counts                    = sample_df[self.text_column].apply(get_pos_counts)
            pos_df                        = pd.DataFrame(pos_counts.tolist())
            pos_df[self.sentiment_column] = sample_df[self.sentiment_column]
            
            # Create visualization
            pos_melted                    = pos_df.melt(id_vars    = [self.sentiment_column],
                                                        var_name   = 'POS',
                                                        value_name = 'Count')
            
            plt.figure(figsize = (12, 6))
            sns.boxplot(x    = 'POS', 
                        y    = 'Count', 
                        hue  = self.sentiment_column, 
                        data = pos_melted)
            
            plt.xticks(rotation = 45)
            plt.title('Distribution of Parts of Speech by Sentiment')
    
            self.save_plot(plt, 'pos_distribution')
            plt.close()
            
        except Exception as e:
            raise
            
            
    def analyze_readability(self) -> None:
        """
        Analyze text readability metrics
        """
        try:
            def calculate_readability(text: str):
                words               = text.split()
                sentences           = text.split('.')
                
                # Rough approximation
                syllables           = sum([len(word)/3 for word in words])  
                
                # Calculate metrics
                avg_word_length     = np.mean([len(word) for word in words])
                avg_sentence_length = len(words) / len(sentences)
                flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * (syllables / len(words))
                
                return {'avg_word_length'     : avg_word_length,
                        'avg_sentence_length' : avg_sentence_length,
                        'flesch_reading_ease' : flesch_reading_ease
                       }
            
            # Calculate readability metrics
            readability_scores                    = self.df[self.text_column].apply(calculate_readability)
            readability_df                        = pd.DataFrame(readability_scores.tolist())
            readability_df[self.sentiment_column] = self.df[self.sentiment_column]
            
            # Create visualizations
            fig, axes                             = plt.subplots(nrows   = 3, 
                                                                 ncols   = 1, 
                                                                 figsize = (10, 15))
            
            for idx, metric in enumerate(readability_df.columns[:-1]):
                sns.boxplot(x    = self.sentiment_column,
                            y    = metric,
                            data = readability_df,
                            ax   = axes[idx])
                
                axes[idx].set_title(f'{metric.replace("_", " ").title()} by Sentiment')
                
            plt.tight_layout()
            self.save_plot(plt, 'readability_metrics')
            plt.close()
            
        except Exception as e:
            raise
         
        
    def run_full_eda(self) -> None:
        """
        Run all EDA analyses
        
        """
        analyses = [self.text_length_analysis,
                    self.word_frequency_analysis,
                    self.generate_wordclouds,
                    self.sentiment_intensity_analysis,
                    self.pos_distribution_analysis,
                    self.analyze_readability
                   ]
        
        for analysis in analyses:
            print(f"Running {analysis.__name__}...")
            analysis()
            
        print("EDA pipeline completed successfully")
        
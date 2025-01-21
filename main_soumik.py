
# Dependencies
import numpy as np
import pandas as pd
from config import DATA_PATH, TEST_DATA_PATH, BATCH_SIZE
from scipy.sparse import hstack
from sklearn.utils import shuffle
from src.helper import batch_generator
from src.data_loader import load_csv_data
from src.text_preprocessor import TextPreprocessor
from src.feature_selector import TextFeatureSelector
from src.sentiment_analyzer import SentimentAnalyzer
from sklearn.model_selection import train_test_split
from src.exploratory_data_analyzer import SentimentEDA
from src.word_level_feature_engineering import TextFeatureEngineering
from src.sentiment_analysis_feature import SentimentFeatureEngineering
from config import (
    SENTIMENT_ANALYSIS_SVM_RBF_RESULT,
    SENTIMENT_ANALYSIS_LOGISTIC_RESULT,
    SENTIMENT_ANALYSIS_LIGHTGBM_RESULT,
    SENTIMENT_ANALYSIS_ADABOOST_RESULT,
    SENTIMENT_ANALYSIS_SVM_SIGMOID_RESULT,
    SENTIMENT_ANALYSIS_RANDOM_FOREST_RESULT,
    SENTIMENT_ANALYSIS_GRADIENT_BOOST_RESULT,
    SENTIMENT_ANALYSIS_GAUSSIAN_NAIVE_BAYES_RESULT,
    SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT,
    SENTIMENT_ANALYSIS_LOGISTIC_DECISION_TREE_RESULT,
    SENTIMENT_ANALYSIS_MULTINOMIAL_NAIVE_BAYES_RESULT
)

def main():
    try:
        print("Loading and preprocessing training data...")
        # Load and preprocess training data
        imdb_ratings_data = load_csv_data(filepath=DATA_PATH)
        preprocessor = TextPreprocessor()
        imdb_ratings_data["clean_text"] = imdb_ratings_data["review"].apply(preprocessor.clean_text)

        # Initialize feature engineering classes
        print("Initializing feature engineering...")
        sentiment_eng = SentimentFeatureEngineering(
            texts=imdb_ratings_data['clean_text'].tolist(), 
            batch_size=BATCH_SIZE
        )
        
        # Create sentiment transformer
        print("Creating sentiment transformer...")
        sentiment_transformer = sentiment_eng.SentimentTransformer(
            analyzer=sentiment_eng, 
            batch_size=BATCH_SIZE
        )
        
        # Fit sentiment transformer
        sentiment_transformer.fit(imdb_ratings_data['clean_text'].tolist())
        
        # Transform sentiment features
        sentiment_features = sentiment_transformer.transform(
            imdb_ratings_data['clean_text'].tolist()
        )

        # Initialize word-level feature engineering
        feature_eng = TextFeatureEngineering(
            texts=imdb_ratings_data['clean_text'].tolist(),
            max_features=5000,  # Adjust as needed
            ngram_range=(1, 3)
        )

        print("Creating word-level features...")
        # Create feature matrices
        freq_vectorizer, freq_features = feature_eng.create_frequency_bow()
        tfidf_vectorizer, tfidf_features = feature_eng.create_tfidf()
        std_tfidf_vectorizer, std_tfidf_features = feature_eng.create_standardized_tfidf()
        bm25l_transformer, bm25l_features = feature_eng.create_bm25l()
        bm25_plus_transformer, bm25_plus_features = feature_eng.create_bm25_plus()
        skipgrams_vectorizer, skipgram_features = feature_eng.create_skipgrams()
        pos_ngram_vectorizer, pos_ngram_features = feature_eng.create_positional_ngrams()

        # Combine feature names
        print("Combining feature names...")
        feature_names = (
            sentiment_transformer.get_feature_names() +
            list(freq_vectorizer.get_feature_names_out()) +
            list(tfidf_vectorizer.get_feature_names_out()) +
            list(std_tfidf_vectorizer.get_feature_names_out()) +
            list(bm25l_transformer.count_vectorizer.get_feature_names_out()) +
            list(bm25_plus_transformer.count_vectorizer.get_feature_names_out()) +
            list(skipgrams_vectorizer.get_feature_names_out()) +
            list(pos_ngram_vectorizer.get_feature_names_out())
        )

        # Combine feature matrices
        print("Combining feature matrices...")
        combined_features = hstack([
            sentiment_features,
            freq_features,
            tfidf_features,
            std_tfidf_features,
            bm25l_features,
            bm25_plus_features,
            skipgram_features,
            pos_ngram_features
        ])

        print("Performing feature selection...")
        # Feature selection
        feature_selector = TextFeatureSelector(
            X=combined_features,
            y=imdb_ratings_data['sentiment'].values,
            feature_names=feature_names,
            n_features=None
        )
        chi_square_features, chi_square_scores = feature_selector.chi_square_selection()
        selected_combined_features = combined_features[:, chi_square_features]

        print("Initializing sentiment analyzer...")
        # Initialize sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer(
            X=selected_combined_features,
            y=imdb_ratings_data["sentiment"].values,
            feature_eng=feature_eng,
            vectorizers=(
                freq_vectorizer,
                tfidf_vectorizer,
                std_tfidf_vectorizer,
                bm25l_transformer,
                bm25_plus_transformer,
                skipgrams_vectorizer,
                pos_ngram_vectorizer
            ),
            selected_feature_indices=chi_square_features
        )

        print("Training model...")
        # Train the model
        multi_layer_perceptron_model = sentiment_analyzer.train_model(
            model_type="multilayer_perceptron"
        )

        print("Training Accuracy....")
        # Evaluation

        evaluation_results = sentiment_analyzer.evaluate_model(multi_layer_perceptron_model)
        
        print("Loading and preprocessing test data...")
        # Load and preprocess test data
        test_data = load_csv_data(filepath=TEST_DATA_PATH)
        test_data["clean_text"] = test_data["Text"].apply(preprocessor.clean_text)

        # Transform test sentiment features using the same transformer
        test_sentiment_features = sentiment_transformer.transform(
            test_data['clean_text'].tolist()
        )

        print("Testing model on unseen data...")
        # Prepare preprocessed features for testing
        preprocessed_features = {
            'sentiment_features': test_sentiment_features,
            'freq_features': freq_vectorizer.transform(test_data['clean_text'].tolist()),
            'tfidf_features': tfidf_vectorizer.transform(test_data['clean_text'].tolist()),
            'std_tfidf_features': std_tfidf_vectorizer.transform(test_data['clean_text'].tolist()),
            'bm25l_features': bm25l_transformer.transform(test_data['clean_text'].tolist()),
            'bm25_plus_features': bm25_plus_transformer.transform(test_data['clean_text'].tolist()),
            'skipgram_features': skipgrams_vectorizer.transform(test_data['clean_text'].tolist()),
            'pos_ngram_features': pos_ngram_vectorizer.transform(test_data['clean_text'].tolist())
        }

        # Test the model using the updated method
        predictions, accuracy = sentiment_analyzer.test_on_unseen_data(
            model=multi_layer_perceptron_model,
            unseen_texts=test_data['clean_text'].tolist(),
            unseen_labels=test_data['Sentiment'].tolist(),
            **preprocessed_features
        )

        print("Saving results...")
        # Save results
        results_df = pd.DataFrame({
            'texts': test_data['Text'].tolist(),
            'true_labels': test_data['Sentiment'].tolist(),
            'predicted_labels': predictions.tolist()
        })
        results_df.to_csv(SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT, index=False)
        
        print(f"Results saved to: {SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT}")
        print(f"Final accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

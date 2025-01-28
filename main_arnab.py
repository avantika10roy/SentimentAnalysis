import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from config import DATA_PATH, TEST_DATA_PATH
from src.feature_selector import TextFeatureSelector
from src.data_loader import load_csv_data
from src.text_preprocessor import TextPreprocessor
from src.character_level_feature_engineering import CharacterLevelFeatureEngineering
from src.word_level_feature_engineering import TextFeatureEngineering
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix



# Logistic Regression Model only on Character Level Features    
class CharacterLevelAnalyzer:
    def __init__(self, data_path, test_data_path):
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.model = LogisticRegression(max_iter=5000)
        self.scaler = StandardScaler(with_mean=False)

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        # Load data
        imdb_ratings_data = load_csv_data(filepath=self.data_path)

        # Preprocess text
        preprocessor = TextPreprocessor()
        imdb_ratings_data["clean_text"] = imdb_ratings_data["review"].apply(preprocessor.clean_text)

        # Extract labels
        self.labels = imdb_ratings_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

        # Initialize feature engineering
        self.feature_eng = CharacterLevelFeatureEngineering(
            texts=imdb_ratings_data['clean_text'].tolist(),
            max_features=1000,
            ngram_range=(3, 3)
        )

        # Extract features
        self.binary_ngram_vectorizer, binary_ngram_features = self.feature_eng.create_char_level_ngram_binary()
        self.frequency_ngram_vectorizer, frequency_ngram_features = self.feature_eng.create_char_level_frequency()
        word_length_features = self.feature_eng.create_word_length_patterns()
        char_type_ratios = self.feature_eng.create_character_type_ratios()
        word_shape_features = self.feature_eng.create_word_shape_features()

        # Combine features
        self.combined_features = hstack([
            binary_ngram_features,
            frequency_ngram_features,
            word_length_features,
            char_type_ratios,
            word_shape_features
        ])

    def train_model(self):
        print("Training model...")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.combined_features, self.labels, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)
        print("Model training completed.")

        # Evaluate on test data
        y_pred_test = self.model.predict(X_test_scaled)
        print("Test Data Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_test))

    def analyze_unseen_data(self):
        print("Analyzing unseen data...")
        # Load unseen data
        unseen_data = pd.read_csv(self.test_data_path)

        # Preprocess unseen text
        preprocessor = TextPreprocessor()
        unseen_data["clean_text"] = unseen_data["Text"].apply(preprocessor.clean_text)

        # Extract unseen features
        binary_ngram_features_unseen = self.binary_ngram_vectorizer.transform(unseen_data['clean_text'])
        frequency_ngram_features_unseen = self.frequency_ngram_vectorizer.transform(unseen_data['clean_text'])
        word_length_features_unseen = self.feature_eng.create_word_length_patterns(unseen_data['clean_text'])
        char_type_ratios_unseen = self.feature_eng.create_character_type_ratios(unseen_data['clean_text'])
        word_shape_features_unseen = self.feature_eng.create_word_shape_features(unseen_data['clean_text'])

        # Combine unseen features
        combined_features_unseen = hstack([
            binary_ngram_features_unseen,
            frequency_ngram_features_unseen,
            word_length_features_unseen,
            char_type_ratios_unseen,
            word_shape_features_unseen
        ])

        # Scale unseen features
        X_unseen_scaled = self.scaler.transform(combined_features_unseen)

        # Predict on unseen data
        y_pred_unseen = self.model.predict(X_unseen_scaled)

        # Evaluate unseen data
        print("Unseen Data Evaluation:")
        print("Classification Report:")
        print(classification_report(
            unseen_data['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0), y_pred_unseen
        ))



# Logistic Regression Model on Character and Word Level Features ( 
# frequency based word, 
# standard tfidf, 
# bm25+, 
# trigram, 
# word_length patterns, 
# character type ratios )
class ModifiedLogisticAnalyzer:
    def __init__(self, data_path, test_data_path):
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.model = LogisticRegression(max_iter=5000)
        self.scaler = StandardScaler(with_mean=False)

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        # Load data
        imdb_ratings_data = load_csv_data(filepath=self.data_path)

        # Preprocess text
        preprocessor = TextPreprocessor()
        imdb_ratings_data["clean_text"] = imdb_ratings_data["review"].apply(preprocessor.clean_text)

        # Extract labels
        self.labels = imdb_ratings_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
        
        
        # Character Level Features
        self.character_features = CharacterLevelFeatureEngineering(
            texts=imdb_ratings_data['clean_text'].tolist(),
            max_features=2000,
            ngram_range=(3, 3)
        )

        # Word Level Features
        self.word_features = TextFeatureEngineering(
            texts = imdb_ratings_data['clean_text'].tolist(),
            max_features = 10000,
            ngram_range  = (1, 3)
        )
        
        # Extract features
        self.freq_vectorizer, freq_features                 = self.word_features.create_frequency_bow()
        self.std_tfidf_vectorizer, std_tfidf_features       = self.word_features.create_standardized_tfidf()
        self.bm25_plus_transformer, bm25_plus_features           = self.word_features.create_bm25_plus()
        self.binary_ngram_vectorizer, binary_ngram_features = self.character_features.create_char_level_ngram_binary()
        word_length_features = self.character_features.create_word_length_patterns()
        char_type_ratios = self.character_features.create_character_type_ratios()

        # Combine features
        self.combined_features = hstack([
            binary_ngram_features,
            word_length_features,
            char_type_ratios,
            freq_features,
            std_tfidf_features, 
            bm25_plus_features,
        ])
        
        feature_selector = TextFeatureSelector( X = self.combined_features,
                                                y = imdb_ratings_data['sentiment'].values)
        
        chi_square_features, chi_square_scores    = feature_selector.chi_square_selection()
        
        self.combined_features = self.combined_features.tocsr()
        
        self.selected_combined_features           = self.combined_features[:, chi_square_features]
        
    def train_model(self):
        print("Training model...")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.combined_features, self.labels, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)
        print("Model training completed.")

        # Evaluate on test data
        y_pred_test = self.model.predict(X_test_scaled)
        print("Test Data Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_test))
        
        
        

    def analyze_unseen_data(self):
        print("Analyzing unseen data...")
        # Load unseen data
        unseen_data = pd.read_csv(self.test_data_path)

        # Preprocess unseen text
        preprocessor = TextPreprocessor()
        unseen_data["clean_text"] = unseen_data["Text"].apply(preprocessor.clean_text)
        
        
        # Extract unseen features
        unseen_word_freq_features = self.freq_vectorizer.transform(unseen_data['clean_text'])
        unseen_std_tfidf_features = self.std_tfidf_vectorizer.transform(unseen_data['clean_text'])
        unseen_bm25_plus_features = self.bm25_plus_transformer.transform(unseen_data['clean_text'])
        unseen_binary_ngram_features = self.binary_ngram_vectorizer.transform(unseen_data['clean_text'])
        unseen_word_length_features = self.character_features.create_word_length_patterns(unseen_data['clean_text'])
        unseen_char_type_ratios_features = self.character_features.create_character_type_ratios(unseen_data['clean_text'])

        
        # pass through spar
        
        
        # Combine unseen features
        combined_features_unseen = hstack([
            unseen_word_freq_features,
            unseen_std_tfidf_features,
            unseen_bm25_plus_features,
            unseen_binary_ngram_features,
            unseen_word_length_features,
            unseen_char_type_ratios_features
        ])

        # Scale unseen features
        X_unseen_scaled = self.scaler.transform(combined_features_unseen)

        # Predict on unseen data
        y_pred_unseen = self.model.predict(X_unseen_scaled)

        # Evaluate unseen data
        print("Unseen Data Evaluation:")
        print("Classification Report:")
        print(classification_report(
            unseen_data['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0), y_pred_unseen
        ))



# Naive Bayes Model on Character and Word Level Features (
# frequency based word, 
# standard tfidf, 
# bm25+, 
# trigram, 
# word_length patterns, 
# character type ratios )
class ModifiedNaiveBayesAnalyzer:
    def __init__(self, data_path, test_data_path):
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.model = MultinomialNB()
        self.scaler = StandardScaler(with_mean=False)

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        # Load data
        imdb_ratings_data = load_csv_data(filepath=self.data_path)

        # Preprocess text
        preprocessor = TextPreprocessor()
        imdb_ratings_data["clean_text"] = imdb_ratings_data["review"].apply(preprocessor.clean_text)

        # Extract labels
        self.labels = imdb_ratings_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
        
        # Character Level Features
        self.character_features = CharacterLevelFeatureEngineering(
            texts=imdb_ratings_data['clean_text'].tolist(),
            max_features=1000,
            ngram_range=(3, 3)
        )

        # Word Level Features
        self.word_features = TextFeatureEngineering(
            texts = imdb_ratings_data['clean_text'].tolist(),
            max_features = 6000,
            ngram_range  = (1, 3)
        )
        
        # Extract features
        self.freq_vectorizer, freq_features                 = self.word_features.create_frequency_bow()
        self.std_tfidf_vectorizer, std_tfidf_features       = self.word_features.create_standardized_tfidf()
        self.bm25_plus_transformer, bm25_plus_features           = self.word_features.create_bm25_plus()
        self.binary_ngram_vectorizer, binary_ngram_features = self.character_features.create_char_level_ngram_binary()
        word_length_features = self.character_features.create_word_length_patterns()
        char_type_ratios = self.character_features.create_character_type_ratios()

        # Combine features
        self.combined_features = hstack([
            binary_ngram_features,
            word_length_features,
            char_type_ratios,
            freq_features,
            std_tfidf_features, 
            bm25_plus_features,
        ])
        
        
        
        feature_selector = TextFeatureSelector( X = self.combined_features,
                                                y = imdb_ratings_data['sentiment'].values)
        
        chi_square_features, chi_square_scores    = feature_selector.chi_square_selection()
        
        self.combined_features = self.combined_features.tocsr()
        
        self.selected_combined_features           = self.combined_features[:, chi_square_features]
        
        
        
        
    def train_model(self):
        print("Training model...")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.selected_combined_features, self.labels, test_size=0.2, random_state=42
        )

        # Train the model
        self.model.fit(X_train, y_train)
        print("Model training completed.")

        # Evaluate on test data
        y_pred_test = self.model.predict(X_test)
        print("Test Data Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_test))

    def analyze_unseen_data(self):
        print("Analyzing unseen data...")
        # Load unseen data
        unseen_data = pd.read_csv(self.test_data_path)

        # Preprocess unseen text
        preprocessor = TextPreprocessor()
        unseen_data["clean_text"] = unseen_data["Text"].apply(preprocessor.clean_text)
        
        
        # Extract unseen features
        unseen_word_freq_features = self.freq_vectorizer.transform(unseen_data['clean_text'])
        unseen_std_tfidf_features = self.std_tfidf_vectorizer.transform(unseen_data['clean_text'])
        unseen_bm25_plus_features = self.bm25_plus_transformer.transform(unseen_data['clean_text'])
        unseen_binary_ngram_features = self.binary_ngram_vectorizer.transform(unseen_data['clean_text'])
        unseen_word_length_features = csr_matrix(self.character_features.create_word_length_patterns(unseen_data['clean_text']))
        unseen_char_type_ratios_features = csr_matrix(self.character_features.create_character_type_ratios(unseen_data['clean_text']))

        # Combine unseen features
        combined_features_unseen = hstack([
            unseen_word_freq_features,
            unseen_std_tfidf_features,
            unseen_bm25_plus_features,
            unseen_binary_ngram_features,
            unseen_word_length_features,
            unseen_char_type_ratios_features
        ])
        
        

        # Predict on unseen data
        y_pred_unseen = self.model.predict(combined_features_unseen)

        # Evaluate unseen data
        print("Unseen Data Evaluation:")
        print("Classification Report:")
        print(classification_report(
            unseen_data['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0), y_pred_unseen
        ))
        

if __name__ == "__main__":
    
    # character_level_model = CharacterLevelAnalyzer(DATA_PATH, TEST_DATA_PATH)
    # character_level_model.load_and_preprocess_data()
    # character_level_model.train_model()
    # character_level_model.analyze_unseen_data()
    
    
    modified_logistic_model = ModifiedLogisticAnalyzer(DATA_PATH, TEST_DATA_PATH)
    modified_logistic_model.load_and_preprocess_data()
    modified_logistic_model.train_model()
    modified_logistic_model.analyze_unseen_data()
    
    
    # modified_nb_model = ModifiedNaiveBayesAnalyzer(DATA_PATH, TEST_DATA_PATH)
    # modified_nb_model.load_and_preprocess_data()
    # modified_nb_model.train_model()
    # modified_nb_model.analyze_unseen_data()
    
# main.py
from src.syntactic_feature_engineering import SentimentAnalysis
from scipy.sparse import hstack

if __name__ == "__main__":
    # Use a subset of 5000 samples from the IMDB dataset
    sentiment_analysis = SentimentAnalysis('./data/IMDB_Dataset.csv', './data/test_data.csv', subset_size=49999)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, X_train_text, X_test_text, X_train_pos, X_test_pos, X_train_dep, X_test_dep, X_train_tree, X_test_tree = sentiment_analysis.load_and_preprocess_data()
    
    # Feature selection
    # Call feature_selection with the proper data
    X_train_pos_selected, X_train_dep_selected, X_train_tree_selected, X_test_pos_selected, X_test_dep_selected, X_test_tree_selected = sentiment_analysis.feature_selection(
        X_train_pos, X_train_dep, X_train_tree, y_train, X_test_pos, X_test_dep, X_test_tree
    )

    # Combine features
    X_train_combined = hstack([X_train_text, X_train_pos_selected, X_train_dep_selected, X_train_tree_selected])
    X_test_combined = hstack([X_test_text, X_test_pos_selected, X_test_dep_selected, X_test_tree_selected])

    # Train and evaluate model
    sentiment_analysis.train_and_evaluate_model(X_train_combined, y_train, X_test_combined, y_test)

    # Evaluate on test data
    sentiment_analysis.evaluate_on_test_data('./data/test_data.csv')

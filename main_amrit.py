# Dependencies
import numpy as np
import pandas as pd
from config import DATA_PATH, TEST_DATA_PATH, SENTIMEANT_ANALYSIS_SVM_POLYNOMIAL_RESULT,SENTIMEANT_ANALYSIS_NAIVE_BYAYES_RESULT
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.utils import shuffle
from src.helper import batch_generator
from src.data_loader import load_csv_data
from src.text_preprocessor import TextPreprocessor
from src.feature_selector import TextFeatureSelector
from src.sentiment_analyzer import SentimentAnalyzer
from src.classification_feature_engineering import ClassFeatureEngineering

# Load the data
data = load_csv_data(filepath=DATA_PATH)
imdb_ratings_data = data.head(10000)
# Preprocess the text
preprocessor = TextPreprocessor()
imdb_ratings_data["clean_text"] = imdb_ratings_data["review"].apply(preprocessor.clean_text)

# Feature Engineering
feature_eng = ClassFeatureEngineering(
    texts=imdb_ratings_data['clean_text'].tolist(),
    labels=imdb_ratings_data['sentiment'].tolist()
)

cs_vectorizer, cs_sparse_matrix = feature_eng.class_specific_vocabulary()
la_vectorizer, la_sparse_matrix = feature_eng.label_aware_embeddings()
mul_vectorizer, mul_sparse_matrix = feature_eng.multi_label_features()

# Validate dimensions of feature matrices
print(f"cs_sparse_matrix shape: {cs_sparse_matrix.shape}")
print(f"la_sparse_matrix shape: {la_sparse_matrix.shape}")
print(f"mul_sparse_matrix shape: {mul_sparse_matrix.shape}")

# Align dimensions of feature matrices
max_rows = max(cs_sparse_matrix.shape[0], la_sparse_matrix.shape[0], mul_sparse_matrix.shape[0])

def align_matrix_rows(matrix, target_rows):
    if matrix.shape[0] < target_rows:
        padding = csr_matrix((target_rows - matrix.shape[0], matrix.shape[1]))
        return vstack([matrix, padding])
    return matrix

cs_sparse_matrix = align_matrix_rows(cs_sparse_matrix, max_rows)
la_sparse_matrix = align_matrix_rows(la_sparse_matrix, max_rows)
mul_sparse_matrix = align_matrix_rows(mul_sparse_matrix, max_rows)

# Combine feature matrices
combined_features = hstack([cs_sparse_matrix, la_sparse_matrix, mul_sparse_matrix])

# Combine feature names
feature_names = (
    list(cs_vectorizer.get_feature_names_out()) +
    list(la_vectorizer.get_feature_names_out()) +
    list(mul_vectorizer.get_feature_names_out())
)

# Feature Selection
feature_selector = TextFeatureSelector(
    X=combined_features,
    y=imdb_ratings_data['sentiment'].values,
    feature_names=feature_names,
    n_features=25000
)

chi_square_features, _ = feature_selector.chi_square_selection()
selected_combined_features = combined_features[:, chi_square_features]

# Sentiment Analysis
sentiment_analyzer = SentimentAnalyzer(
    X=selected_combined_features,
    y=imdb_ratings_data["sentiment"].values,
    feature_eng=feature_eng,
    vectorizers=(cs_vectorizer, la_vectorizer, mul_vectorizer),
    selected_feature_indices=chi_square_features
)

svm_polynomial_model = sentiment_analyzer.train_model(model_type="svm", kernel='poly')
evaluation_results = sentiment_analyzer.evaluate_model(svm_polynomial_model)

# Predict on unseen data
test_data = load_csv_data(filepath=TEST_DATA_PATH)
logistic_predictions, unseen_accuracy = sentiment_analyzer.test_on_unseen_data(
    model=svm_polynomial_model,
    unseen_texts=list(test_data['Text']),
    unseen_labels=list(test_data['Sentiment']),
    cs_features=cs_vectorizer.transform(test_data['Text']),
    la_features=la_vectorizer.transform(test_data['Text']),
    mul_features=mul_vectorizer.transform(test_data['Text'])
)

# Save predictions to a CSV file
all_test_data = {
    'texts': list(test_data['Text']),
    'true_labels': list(test_data['Sentiment']),
    'predicted_labels': logistic_predictions
}

svm_polynomial_predictions_df = pd.DataFrame.from_dict(data=all_test_data)
svm_polynomial_predictions_df.to_csv(path_or_buf=SENTIMEANT_ANALYSIS_SVM_POLYNOMIAL_RESULT, index=False)

print(f"Sentiment Analysis results saved to: {SENTIMEANT_ANALYSIS_SVM_POLYNOMIAL_RESULT}")



naive_bayes_model = sentiment_analyzer.train_model(model_type="naive_bayes")
evaluation_results = sentiment_analyzer.evaluate_model(naive_bayes_model)


# Predict on unseen data
test_data = load_csv_data(filepath=TEST_DATA_PATH)
logistic_predictions, unseen_accuracy = sentiment_analyzer.test_on_unseen_data(
    model=naive_bayes_model,
    unseen_texts=list(test_data['Text']),
    unseen_labels=list(test_data['Sentiment']),
    cs_features=cs_vectorizer.transform(test_data['Text']),
    la_features=la_vectorizer.transform(test_data['Text']),
    mul_features=mul_vectorizer.transform(test_data['Text'])
)

# Save predictions to a CSV file
all_test_data = {
    'texts': list(test_data['Text']),
    'true_labels': list(test_data['Sentiment']),
    'predicted_labels': logistic_predictions
}

naive_bayes_predictions_df = pd.DataFrame.from_dict(data=all_test_data)
naive_bayes_predictions_df.to_csv(path_or_buf=SENTIMEANT_ANALYSIS_NAIVE_BYAYES_RESULT, index=False)

print(f"Sentiment Analysis results saved to: {SENTIMEANT_ANALYSIS_NAIVE_BYAYES_RESULT}")

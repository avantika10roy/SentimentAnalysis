import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.document_level_feature_engineering import TextFeatureEngineering_document
from config import DATA_PATH
def main():
    # Load the IMDB dataset
    dataset_path = DATA_PATH
    data = pd.read_csv(dataset_path)

    # Extract the 'review' column and labels
    reviews = data['review'].tolist()
    labels = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).tolist()  # Encode sentiment labels

    # Initialize TextFeatureEngineering_document
    feature_engineer = TextFeatureEngineering_document(reviews, max_features=500)

    # Generate all features
    features = feature_engineer.create_all_features()

    # Combine all features into a single matrix
    #lda_features = features['lda'][1]
    lsi_features = features['lsi'][1]
    #document_embeddings = features['document_embeddings'][1]
    #document_similarity = features['document_similarity']
    hierarchical_features = features['hierarchical_features'][1]

    # Stack all features horizontally
    combined_features = np.hstack((
        #lda_features,
        lsi_features,
        #document_embeddings,
        #document_similarity,
        hierarchical_features.reshape(-1, 1)  # Reshape clusters to 2D
    ))

    # Standardize the features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)

if __name__ == "__main__":
    main()

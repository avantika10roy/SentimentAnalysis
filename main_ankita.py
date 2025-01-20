import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from src.text_preprocessor import TextPreprocessor
from src.sentiment_analysis_feature_ankita import SentimentFeatureEngineering

imdb_ratings_data = pd.read_csv("data/IMDB_Dataset.csv")

preprocessor = TextPreprocessor()
imdb_ratings_data["clean_text"] = imdb_ratings_data["review"].apply(preprocessor.clean_text)

texts = imdb_ratings_data["clean_text"].tolist() 

test_data = pd.read_csv("/Users/itobuz/nlp_project/Sentiment_Analysis/data/test_data.csv")

test_data["clean_text"] = test_data["Text"].apply(preprocessor.clean_text)

test_texts = test_data["clean_text"].tolist()
test_labels = test_data["Sentiment"].tolist()

def main():
    sentiment_feature_engineer = SentimentFeatureEngineering(texts, max_features=500, ngram_range=(1, 2))
    
    X, y = sentiment_feature_engineer.prepare_data_for_training()

    sentiment_feature_engineer_test = SentimentFeatureEngineering(test_texts, max_features=500, ngram_range=(1, 2))
    X_test, y_test = sentiment_feature_engineer_test.prepare_data_for_training()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = [
        (LogisticRegression(max_iter=1000), 'Logistic Regression'),
        (SVC(kernel='linear'), 'SVM'),
        (GaussianNB(), 'Gaussian Naive Bayes'),
        (XGBClassifier(random_state=42), 'XGBoost'),
        (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42), 'Multilayer Perceptron')
    ]

    accuracy_results = []

    sentiment_results = []

    for model, name in tqdm(models, desc="Training Models", unit="model"):
        with tqdm(total=1, desc=f'{name} Training', position=0, leave=True) as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracy_results.append((name, accuracy * 100))
        
        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Test Accuracy for {name}: {test_accuracy * 100:.2f}%")

    accuracy_results.append((name, accuracy * 100, test_accuracy * 100))
        
    results = pd.DataFrame(accuracy_results, columns=['Model', 'Accuracy'])
    results.to_csv('model_results.csv', index=False)
    print("Results saved to model_results.csv")

if __name__ == "__main__":
    main()

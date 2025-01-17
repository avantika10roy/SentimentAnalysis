import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Import all required models
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                            GradientBoostingClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SentimentFeatureExtractor:
    """
    A class for extracting sentiment features from text data
    """
    def __init__(self, sentiment_analyzer):
        """
        Initialize the feature extractor with a sentiment analyzer

        Arguments:
        ----------
            sentiment_analyzer: Instance of OptimizedSentimentFeatures class
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.feature_names = []
        
    def extract_features(self, texts, batch_size=1000):
        """
        Extract sentiment features from texts

        Arguments:
        ----------
            texts: List of text documents
            batch_size: Number of documents to process at once

        Returns:
        --------
            Sparse matrix of sentiment features
        """
        # Sentiment analysis results in batches
        results = self.sentiment_analyzer.batch_analyze_sentiment(texts, batch_size)
        
        # Initializing dictionary to store feature values
        feature_dict = defaultdict(list)
        
        # Process each result and extract features
        for result in results:
            # VADER features
            vader = result['vader_sentiment']
            for k, v in vader.items():
                feature_dict[f'vader_{k}'].append(v)
            
            # Emotion counts
            emotions = result['emotion_counts']
            for emotion, count in emotions.items():
                feature_dict[f'emotion_{emotion}'].append(count)
            
            # Aspect sentiments
            aspects = result['aspect_based_sentiment']
            if aspects:
                sentiments = list(aspects.values())
                feature_dict['aspect_mean'].append(np.mean(sentiments))
                feature_dict['aspect_min'].append(min(sentiments))
                feature_dict['aspect_max'].append(max(sentiments))
            else:
                feature_dict['aspect_mean'].append(0)
                feature_dict['aspect_min'].append(0)
                feature_dict['aspect_max'].append(0)
            
            # Polarity patterns
            polarity = result['polarity_patterns']
            for k, v in polarity.items():
                if k != 'overall_sentiment':
                    feature_dict[f'polarity_{k}'].append(v)
        
        # Convert to sparse matrix
        self.feature_names = list(feature_dict.keys())
        rows = len(results)
        cols = len(self.feature_names)
        data = []
        row_ind = []
        col_ind = []
        
        for col_idx, feature in enumerate(self.feature_names):
            for row_idx, value in enumerate(feature_dict[feature]):
                if value != 0:
                    data.append(value)
                    row_ind.append(row_idx)
                    col_ind.append(col_idx)
        
        return csr_matrix((data, (row_ind, col_ind)), shape=(rows, cols))

    def get_feature_names(self):
        """
        Returns the names of the features extracted by the sentiment analyzer.
        """
        return self.feature_names

    def transform(self, texts, batch_size=1000):
        """
        Transform the input texts into sentiment features.

        Arguments:
        ----------
            texts: List of text documents
            batch_size: Number of documents to process at once

        Returns:
        --------
            Sparse matrix of sentiment features
        """
        return self.extract_features(texts, batch_size)


class EnhancedSentimentAnalyzer:
    """
    A class for training and evaluating sentiment analysis models with multiple feature types
    """
    def __init__(self, X, y, feature_extractors, selected_feature_indices=None, test_size=0.2, random_state=42):
        """
        Initialize the sentiment analyzer

        Arguments:
        ----------
            X: Combined feature matrix (sparse matrix or ndarray)
            y: Target labels
            feature_extractors: Dictionary of feature extractors
            selected_feature_indices: Indices of selected features
            test_size: Proportion of test data
            random_state: Random seed
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.feature_extractors = feature_extractors
        self.selected_feature_indices = selected_feature_indices
        self.scaler = StandardScaler(with_mean=False)
        
        # Scaling features 
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self, model_type="logistic_regression", kernel=None, **kwargs):
        """
        Train a sentiment analysis model

        Arguments:
        ----------
            model_type: Type of model to train
            kernel: Kernel type for SVM
            kwargs: Additional model parameters

        Returns:
        --------
            Trained model
        """
        models = {
            "logistic_regression": lambda: LogisticRegression(max_iter=1000, **kwargs),
            "svm": lambda: SVC(kernel=kernel or "rbf", **kwargs),
            "random_forest": lambda: RandomForestClassifier(**kwargs),
            "gaussian_naive_bayes": lambda: GaussianNB(),
            "multinomial_naive_bayes": lambda: MultinomialNB(**kwargs),
            "adaboost": lambda: AdaBoostClassifier(**kwargs),
            "gradient_boost": lambda: GradientBoostingClassifier(**kwargs),
            "lightgbm": lambda: LGBMClassifier(**kwargs),
            "multilayer_perceptron": lambda: MLPClassifier(hidden_layer_sizes=(1000,), max_iter=1000, **kwargs),
            "logistic_decision_tree": lambda: StackingClassifier(
                estimators=[('decision_tree', DecisionTreeClassifier(max_depth=50, min_samples_split=10, min_samples_leaf=5, **kwargs))],
                final_estimator=LogisticRegression(max_iter=1000, penalty='l2', C=1.0, solver='lbfgs', **kwargs),
                stack_method='predict_proba'
            )
        }

        if model_type not in models:
            raise ValueError(f"Unsupported model_type. Choose from: {', '.join(models.keys())}")

        print(f"Training {model_type}...")
        model = models[model_type]()

        # Using GaussianNB
        if isinstance(model, GaussianNB):
            X_train = self.X_train.toarray() if issparse(self.X_train) else self.X_train
        else:
            X_train = self.X_train

        model.fit(X_train, self.y_train)
        return model

    def evaluate_model(self, model):
        """
        Evaluate the trained model

        Arguments:
        ----------
            model: Trained model

        Returns:
        --------
            Dictionary of evaluation metrics
        """
        print("Evaluating model...")
        
        if isinstance(model, GaussianNB):
            X_test = self.X_test.toarray() if issparse(self.X_test) else self.X_test
        else:
            X_test = self.X_test

        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }

    def test_on_unseen_data(self, model, unseen_texts, unseen_labels=None):
        """
        Test the model on unseen data

        Arguments:
        ----------
            model: Trained model
            unseen_texts: List of unseen text data
            unseen_labels: True labels for unseen data (optional)

        Returns:
        --------
            Predictions and accuracy (if labels provided)
        """
        print("Processing unseen data...")

        # Extract features for unseen data using all feature extractors
        unseen_features = []
        for extractor in self.feature_extractors.values():
            features = extractor.extract_features(unseen_texts)
            unseen_features.append(features)

        # Combine all features
        unseen_combined = hstack(unseen_features)

        # Apply feature selection if available
        if self.selected_feature_indices is not None:
            unseen_combined = unseen_combined[:, self.selected_feature_indices]

        # Scale features
        unseen_combined = self.scaler.transform(unseen_combined)

        # Convert to dense if using GaussianNB
        if isinstance(model, GaussianNB):
            unseen_combined = unseen_combined.toarray() if issparse(unseen_combined) else unseen_combined

        # Make predictions
        predictions = model.predict(unseen_combined)

        # Print predictions
        print("Predictions on Unseen Data:")
        for text, pred in zip(unseen_texts, predictions):
            print(f"Text: {text}\nPredicted Sentiment: {pred}\n")

        # Calculate accuracy if labels are provided
        if unseen_labels is not None:
            if len(unseen_labels) != len(predictions):
                raise ValueError("Number of unseen_labels must match number of predictions")
            
            accuracy = accuracy_score(unseen_labels, predictions)
            print(f"Accuracy on Unseen Data: {accuracy:.4f}")
            return predictions, accuracy

        return predictions
    
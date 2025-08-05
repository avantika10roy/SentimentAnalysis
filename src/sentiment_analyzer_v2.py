# -- Dependencies -- 
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SentimentAnalyzer:
    """
    A class for training and evaluating sentiment analysis models, including testing on unseen data.
    """

    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initialize the SentimentAnalyzer by splitting the data.

        Arguments:
        ----------
            X            : Feature matrix (ndarray)
            y            : Target labels (array-like)
            test_size    : Proportion of data to use for testing (default: 0.2)
            random_state : Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train_model(self, model_type: str = "logistic_regression", kernel=None, **kwargs):
        """
        Train a sentiment analysis model.

        Arguments:
        ----------
            model_type : Type of model to train (e.g., "logistic_regression", "svm", "random_forest", etc.)
            kernel     : Kernel type for SVM (e.g., "linear", "poly", "rbf", "sigmoid")
            kwargs     : Additional arguments for model initialization

        Returns:
        --------
            Trained model
        """
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, **kwargs)
        elif model_type == "svm":
            kernel = kernel or "rbf"
            model = SVC(kernel=kernel, **kwargs)
        elif model_type == "random_forest":
            model = RandomForestClassifier(**kwargs)
        elif model_type == "gradient_boosting":
            model = LGBMClassifier(**kwargs)
        elif model_type == "mlp":
            model = MLPClassifier(max_iter=500, **kwargs)
        elif model_type == "naive_bayes":
            model = MultinomialNB(**kwargs)
        elif model_type == "knn":
            model = KNeighborsClassifier(**kwargs)
        else:
            raise ValueError(
                "Unsupported model_type. Choose from: 'logistic_regression', 'svm', 'random_forest', "
                "'gradient_boosting', 'mlp', 'naive_bayes', 'knn'."
            )

        print(f"Training {model_type}...")
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate_model(self, model):
        """
        Evaluate a trained model on the test set.

        Arguments:
        ----------
            model : Trained model

        Returns:
        --------
            Dictionary containing evaluation metrics
        """
        print("Evaluating model...")
        y_pred = model.predict(self.X_test)
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
            "confusion_matrix": cm,
        }

    def test_on_unseen_data(self, model, unseen_texts, vectorizer_function, original_class):
        """
        Test the model on unseen data.

        Arguments:
        ----------
            model              : Trained model
            unseen_texts       : List of unseen text data
            vectorizer_function: Function to convert texts into vector embeddings

        Returns:
        --------
            Predictions for the unseen data
        """
        print("Processing unseen data...")

        # Convert unseen texts to vector embeddings using the provided function
        unseen_vectors = vectorizer_function(unseen_texts)

        # Predict sentiments
        predictions = model.predict(unseen_vectors)

        accuracy = accuracy_score(original_class, predictions)
        report = classification_report(original_class, predictions)
        cm = confusion_matrix(original_class, predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            'predictions' : predictions
        }

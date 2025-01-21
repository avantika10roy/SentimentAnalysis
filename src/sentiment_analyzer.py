# Dependencies
from sklearn.svm import SVC
from scipy.sparse import hstack
from scipy.sparse import issparse
from lightgbm import LGBMClassifier
from config import MODEL_PARAMS_DICT
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


# Sentiment Analysis Model Fitting
class SentimentAnalyzer:
    """
    A class for training and evaluating sentiment analysis models, including testing on unseen data
    """

    def __init__(self, X, y, feature_eng, selected_feature_indices, test_size=0.2, random_state=42, vectorizers=None):
        """
        Initialize the SentimentAnalyzer by splitting the data

        Arguments:
        ----------
            X                        : Feature matrix (sparse matrix or ndarray)
            
            y                        : Target labels (array-like)
            
            feature_eng              : Instance of TextFeatureEngineering
            
            vectorizers              : Tuple of vectorizers used for feature transformation
            
            selected_feature_indices : Indices of selected features after feature selection
            
            test_size                : Proportion of data to use for testing (default: 0.2)
            
            random_state             : Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
                                                                                y, 
                                                                                test_size    = test_size, 
                                                                                random_state = random_state)
        
        self.feature_eng                                     = feature_eng
        self.vectorizers                                     = vectorizers
        self.selected_feature_indices                        = selected_feature_indices

        
    def train_model(self, model_type:str = "logistic_regression", kernel:str = None, **kwargs):
        """
        Train a sentiment analysis model

        Arguments:
        ----------
            model_type { str } : Type of model to train (e.g: "logistic_regression", "svm", "random_forest")
            
            kernel     { str } : Kernel type for SVM (e.g., "linear", "poly", "rbf", "sigmoid")
            
            kwargs             : Additional arguments for the model initialization

        Returns:
        --------
            Trained model
        """
        if (model_type == "logistic_regression"):
            model = LogisticRegression(max_iter = MODEL_PARAMS_DICT['max_iter'], 
                                       **kwargs)
            
        elif (model_type == "svm"):
            
            if (kernel is None):
                # Default kernel
                kernel = "rbf"  
                
            model = SVC(kernel = kernel, **kwargs)
            
        elif (model_type == "random_forest"):
            model = RandomForestClassifier(**kwargs)

        elif (model_type == "gaussian_naive_bayes"):
            if issparse(self.X_train):
                self.X_train = self.X_train.toarray()

            model = GaussianNB(**kwargs)
            
        elif (model_type == "multinomial_naive_bayes"):
            model = MultinomialNB(**kwargs)

        elif (model_type == "adaboost"):
            model = AdaBoostClassifier(**kwargs)

        elif (model_type == "gradient_boost"):
            model = GradientBoostingClassifier(**kwargs)

        elif (model_type == "lightgbm"):
            model = LGBMClassifier(**kwargs)

        elif (model_type == 'label_propagation'):
            model = LabelPropagation(kernel      = 'knn',
                                     n_neighbors = MODEL_PARAMS_DICT['n_neighbors'], 
                                     max_iter    = MODEL_PARAMS_DICT['max_iter'], 
                                     tol         = MODEL_PARAMS_DICT['tol'],
                                     **kwargs
                                    )

        elif (model_type == "multilayer_perceptron"):
            model = MLPClassifier(hidden_layer_sizes = MODEL_PARAMS_DICT['hidden_layer_size'], 
                                  max_iter           = MODEL_PARAMS_DICT['max_iter'], 
                                  **kwargs)

        elif (model_type == 'hist_gradient_boosting_classifier'):
            if issparse(self.X_train):
                self.X_train = self.X_train.toarray()

            model = HistGradientBoostingClassifier(loss              = MODEL_PARAMS_DICT['loss'], 
                                                   learning_rate     = MODEL_PARAMS_DICT['learning_rate'], 
                                                   max_iter          = MODEL_PARAMS_DICT['max_iter'],
                                                   min_samples_leaf  = MODEL_PARAMS_DICT['min_samples_leaf'],
                                                   l2_regularization = MODEL_PARAMS_DICT['l2_regularization'],
                                                   max_features      = MODEL_PARAMS_DICT['max_features'],
                                                   **kwargs)


        elif (model_type == "logistic_decision_tree"):
            # Create a logistic regression model
            logistic_model      = LogisticRegression(max_iter = MODEL_PARAMS_DICT['max_iter'], 
                                                     penalty  = MODEL_PARAMS_DICT['penalty'], 
                                                     C        = MODEL_PARAMS_DICT['C'], 
                                                     solver   = MODEL_PARAMS_DICT['solver'],
                                                     **kwargs)

            # Create a decision tree model
            decision_tree_model = DecisionTreeClassifier(max_depth         = MODEL_PARAMS_DICT['max_depth'], 
                                                         min_samples_split = MODEL_PARAMS_DICT['min_samples_split'], 
                                                         min_samples_leaf  = MODEL_PARAMS_DICT['min_samples_leaf'],
                                                         **kwargs)

            # Combine them in a stacking model
            model               = StackingClassifier(estimators      = [('decision_tree', decision_tree_model)], 
                                                     final_estimator = logistic_model, 
                                                     stack_method    = 'predict_proba',
                                                     **kwargs)
        
        elif (model_type == "logistic_gaussian_naive_bayes"):
            # Create a logistic regression model
            logistic_model = LogisticRegression(max_iter = MODEL_PARAMS_DICT['max_iter'], 
                                                penalty  = MODEL_PARAMS_DICT['penalty'], 
                                                C        = MODEL_PARAMS_DICT['C'], 
                                                solver   = MODEL_PARAMS_DICT['solver'],
                                                **kwargs)

            # Gaussian Naive Bayes does not work with sparse matrices, so convert to dense if needed
            if issparse(self.X_train):
                self.X_train = self.X_train.toarray()

            # Create Gaussian Naive Bayes model
            gaussian_naive_bayes = GaussianNB()

            # Combine them in a stacking model (Logistic Regression as base model, Gaussian Naive Bayes as final estimator)
            model                = StackingClassifier(estimators      = [('logistic_regression', logistic_model)], 
                                                      final_estimator = gaussian_naive_bayes, 
                                                      stack_method    = 'predict_proba',
                                                      **kwargs)
        
        else:
            raise ValueError("Unsupported model_type. Choose from : 'logistic_regression', 'svm', 'random_forest', 'multinomial_naive_bayes', \
                             'gaussian_naive_bayes', 'adaboost', 'gradient_boost', 'lightgbm', 'logistic_decision_tree', 'logistic_gaussian_naive_bayes', 'multilayer_perceptron".replace('  ', ''))

        print(f"Training {model_type}...")
        model.fit(self.X_train, self.y_train)

        return model

    def evaluate_model(self, model):
        """
        Evaluate a trained model on the test set

        Arguments:
        ----------
            model : Trained model

        Returns:
        --------
            Dictionary containing evaluation metrics
        """
        print("Evaluating model...")

        if (isinstance(model, StackingClassifier)):
            if (isinstance(model.final_estimator_, GaussianNB)):
                # Handle dense conversion for GaussianNB final estimator in stacking model
                X_test_dense = self.X_test.toarray() if hasattr(self.X_test, "toarray") else self.X_test
                y_pred        = model.predict(X_test_dense)
        
        elif ((isinstance(model, GaussianNB)) or (isinstance(model, HistGradientBoostingClassifier))):
            # Directly handle GaussianNB or HistGradientBoostingClassifier
            X_test_dense = self.X_test.toarray() if hasattr(self.X_test, "toarray") else self.X_test
            y_pred       = model.predict(X_test_dense)
        
        else:
            y_pred = model.predict(self.X_test)
            
        accuracy = accuracy_score(self.y_test, y_pred)
        report   = classification_report(self.y_test, y_pred)
        cm       = confusion_matrix(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

        return {"accuracy"              : accuracy,
                "classification_report" : report,
                "confusion_matrix"      : cm,
               }

    
    def test_on_unseen_data(self, model, unseen_texts, unseen_labels=None, **preprocessed_features):
        """
        Test the model on unseen data

        Arguments:
        ----------
            model                 : Trained model
            
            unseen_texts          : List of unseen text data

            unseen_labels         : True labels for the unseen data

            preprocessed_features : Preprocessed feature matrices (e.g., binary_features, tfidf_features, bm25_features, etc.)

        Returns:
        --------
            Predictions for the unseen data
        """
        print("Processing unseen data...")

        # Dynamically combine all passed feature matrices
        unseen_combined_features = hstack([preprocessed_features[key] for key in preprocessed_features])

        # Select features using the indices chosen during feature selection
        unseen_selected_features = unseen_combined_features[:, self.selected_feature_indices]

        # Convert unseen features to dense for Gaussian Naive Bayes
        if ((isinstance(model, GaussianNB)) or (isinstance(model, HistGradientBoostingClassifier))):
            unseen_selected_features = unseen_selected_features.toarray() if hasattr(unseen_selected_features, "toarray") else unseen_selected_features
        
        elif (isinstance(model, StackingClassifier)):
            if (isinstance(model.final_estimator_, GaussianNB)):
                unseen_selected_features = unseen_selected_features.toarray() if hasattr(unseen_selected_features, "toarray") else unseen_selected_features

        # Predict sentiments
        predictions              = model.predict(unseen_selected_features)

        # Print predictions
        print("Predictions on Unseen Data:")
        for text, pred in zip(unseen_texts, predictions):
            print(f"Text: {text}\nPredicted Sentiment: {pred}\n")

        # Compute accuracy if unseen_labels are provided
        if unseen_labels is not None:
            print(f"Number of unseen_labels: {len(unseen_labels)}")

            if (len(unseen_labels) != len(predictions)):
                raise ValueError("The number of unseen_labels must match the number of predictions.")
                
            accuracy = accuracy_score(unseen_labels, predictions)
            print(f"Accuracy on Unseen Data : {accuracy:.4f}")
            return predictions, accuracy

        return predictions
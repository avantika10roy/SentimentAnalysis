import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from scipy.sparse import hstack
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class SentimentAnalysis:
    def __init__(self, train_data_path, test_data_path, subset_size=None):
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load datasets
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
        # Initialize vectorizers
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.vectorizer_pos = TfidfVectorizer(max_features=5000)
        self.vectorizer_dep = TfidfVectorizer(max_features=5000)
        self.vectorizer_tree = TfidfVectorizer(max_features=5000)
        
        # Initialize feature selectors (these will be fitted later)
        self.selector_pos = None
        self.selector_dep = None
        self.selector_tree = None
        
        # Initialize base models
        nb_model = GaussianNB()
        dt_model = DecisionTreeClassifier()
        
        # Initialize meta-model
        lr_model = LogisticRegression()
        
        # Initialize stacking model
        self.stacking_model = StackingClassifier(
            estimators=[('nb', nb_model), ('dt', dt_model)],
            final_estimator=lr_model
        )
        self.subset_size = subset_size
        
    def extract_syntactic_features(self, text):
        """Extract syntactic features (POS tags, dependencies, parse tree heads)."""
        doc = self.nlp(text)
        pos_tags = [token.pos_ for token in doc]  # POS tags
        dependencies = [token.dep_ for token in doc]  # Dependency relationships
        parse_tree = [token.head.pos_ for token in doc]  # Parse tree heads
        return ' '.join(pos_tags), ' '.join(dependencies), ' '.join(parse_tree)
    
    def load_and_preprocess_data(self):
        """Load and preprocess data (train and test)."""
    # Load the training data
        imdb_data = pd.read_csv(self.train_data_path)
        X = imdb_data['review']  # Assuming reviews are in 'review' column
        y = imdb_data['sentiment']  # Assuming sentiment labels are in 'sentiment' column
        y = y.map({'positive': 1, 'negative': 0})  # Convert sentiments to numeric labels
        
        # Use a subset if required
        if hasattr(self, 'subset_size') and self.subset_size:
            X, _, y, _ = train_test_split(X, y, train_size=self.subset_size, random_state=42)

        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Extract syntactic features
        print("Extracting syntactic features...")
        train_pos, train_dep, train_tree = zip(*[self.extract_syntactic_features(text) for text in tqdm(X_train)])
        test_pos, test_dep, test_tree = zip(*[self.extract_syntactic_features(text) for text in tqdm(X_test)])
        
        # Vectorize text data and syntactic features
        print("Vectorizing original text...")
        X_train_text = self.vectorizer.fit_transform(X_train)
        X_test_text = self.vectorizer.transform(X_test)
        
        print("Vectorizing syntactic features...")
        X_train_pos = self.vectorizer_pos.fit_transform(train_pos)
        X_test_pos = self.vectorizer_pos.transform(test_pos)
        X_train_dep = self.vectorizer_dep.fit_transform(train_dep)
        X_test_dep = self.vectorizer_dep.transform(test_dep)
        X_train_tree = self.vectorizer_tree.fit_transform(train_tree)
        X_test_tree = self.vectorizer_tree.transform(test_tree)
    
        return X_train, X_test, y_train, y_test, X_train_text, X_test_text, X_train_pos, X_test_pos, X_train_dep, X_test_dep, X_train_tree, X_test_tree

               
            

    def feature_selection(self, X_train_pos, X_train_dep, X_train_tree, y_train, X_test_pos, X_test_dep, X_test_tree):
        """Apply feature selection using chi-square test."""
        print("Selecting top features using Chi-Square test...")
        selector_pos = SelectKBest(chi2, k=1000).fit(X_train_pos, y_train)
        selector_dep = SelectKBest(chi2, k=1000).fit(X_train_dep, y_train)
        selector_tree = SelectKBest(chi2, k=1000).fit(X_train_tree, y_train)

        # Ensure feature matrices are 2D
        X_train_pos_selected = selector_pos.transform(X_train_pos)
        X_train_dep_selected = selector_dep.transform(X_train_dep)
        X_train_tree_selected = selector_tree.transform(X_train_tree)

        X_test_pos_selected = selector_pos.transform(X_test_pos)
        X_test_dep_selected = selector_dep.transform(X_test_dep)
        X_test_tree_selected = selector_tree.transform(X_test_tree)

        return X_train_pos_selected, X_train_dep_selected, X_train_tree_selected, X_test_pos_selected, X_test_dep_selected, X_test_tree_selected

    def train_and_evaluate_model(self, X_train_combined, y_train, X_test_combined, y_test):
        """Train the stacking model and evaluate accuracy."""
        print("Training stacking model...")
        
        # Convert sparse matrix to dense format
        X_train_combined_dense = X_train_combined.toarray()
        X_test_combined_dense = X_test_combined.toarray()

        # Train the stacking model
        self.stacking_model.fit(X_train_combined_dense, y_train)
        
        print("Evaluating the model...")
        y_pred = self.stacking_model.predict(X_test_combined_dense)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy, y_pred

    def evaluate_on_test_data(self, test_data_path):
        """Evaluate the trained stacking model on unseen test data."""
        # Load test data
        test_data = pd.read_csv(test_data_path)
        test_X = test_data['Text']  # Assuming reviews are in 'Text' column
        test_y = test_data['Sentiment']  # Assuming sentiment labels are in 'Sentiment' column
        test_y = test_y.map({'positive': 1, 'negative': 0})  # Convert sentiments to numeric labels
        
        # Extract syntactic features for the test set
        test_pos, test_dep, test_tree = zip(*[self.extract_syntactic_features(text) for text in tqdm(test_X)])

        # Vectorize the original test text data
        X_test_text = self.vectorizer.transform(test_X)
        
        # Vectorize syntactic features (POS tags, dependencies, parse tree)
        X_test_pos = self.vectorizer_pos.transform(test_pos)
        X_test_dep = self.vectorizer_dep.transform(test_dep)
        X_test_tree = self.vectorizer_tree.transform(test_tree)
        
        # Apply feature selection (same selectors used for training data)
        X_test_pos_selected, X_test_dep_selected, X_test_tree_selected, _, _, _ = self.feature_selection(
            X_test_pos, X_test_dep, X_test_tree, test_y, X_test_pos, X_test_dep, X_test_tree
        )
        
        # Combine features
        X_test_combined = hstack([X_test_text, X_test_pos_selected, X_test_dep_selected, X_test_tree_selected])

        # Convert sparse matrix to dense format before prediction
        X_test_combined_dense = X_test_combined.toarray()

        # Make predictions on the test data using the trained stacking model
        y_test_pred = self.stacking_model.predict(X_test_combined_dense)

        # Evaluate accuracy
        test_accuracy = accuracy_score(test_y, y_test_pred)
        print(f'Accuracy on test data: {test_accuracy:.4f}')
        print("Classification Report on test data:")
        print(classification_report(test_y, y_test_pred))




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

    # sentiment_analysis.evaluate

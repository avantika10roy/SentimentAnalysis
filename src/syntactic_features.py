#Done by Subhas Mukherjee

import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from scipy.sparse import hstack
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB

class SentimentAnalysis:
    def __init__(self, train_data_path, test_data_path, subset_size=None):
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load datasets
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
        # Initialize vectorizers
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.vectorizer_pos = TfidfVectorizer(max_features=10000)
        self.vectorizer_dep = TfidfVectorizer(max_features=10000)
        self.vectorizer_tree = TfidfVectorizer(max_features=10000)
        
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
        
        
    def extract_syntactic_features(self, texts):
        """Extract syntactic features (POS tags, dependencies, parse tree heads)."""
        pos_tags = []
        dependencies = []
        parse_tree = []
        
        for text in tqdm(texts):
            doc = self.nlp(text)
            pos_tags.append(' '.join([token.pos_ for token in doc]))  # POS tags
            dependencies.append(' '.join([token.dep_ for token in doc]))  # Dependency relationships
            parse_tree.append(' '.join([token.head.pos_ for token in doc]))  # Parse tree heads
        
        return pos_tags, dependencies, parse_tree
    
    def load_and_preprocess_data(self):
        """Load and preprocess data (train and test)."""
        # Load the training data
        imdb_data = pd.read_csv(self.train_data_path)
        X = imdb_data['review']  # Assuming reviews are in 'review' column
        y = imdb_data['sentiment']  # Assuming sentiment labels are in 'sentiment' column
        y = y.map({'positive': 1, 'negative': 0})  # Convert sentiments to numeric labels
        
        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def vectorize_features(self, X_train, X_test, pos_tags_train, pos_tags_test, dep_tags_train, dep_tags_test, tree_tags_train, tree_tags_test):
        """Vectorize text and syntactic features."""
        print("Vectorizing original text...")
        X_train_text = self.vectorizer.fit_transform(X_train)
        X_test_text = self.vectorizer.transform(X_test)
        
        print("Vectorizing syntactic features...")
        X_train_pos = self.vectorizer_pos.fit_transform(pos_tags_train)
        X_test_pos = self.vectorizer_pos.transform(pos_tags_test)
        X_train_dep = self.vectorizer_dep.fit_transform(dep_tags_train)
        X_test_dep = self.vectorizer_dep.transform(dep_tags_test)
        X_train_tree = self.vectorizer_tree.fit_transform(tree_tags_train)
        X_test_tree = self.vectorizer_tree.transform(tree_tags_test)
        
        return X_train_text, X_test_text, X_train_pos, X_test_pos, X_train_dep, X_test_dep, X_train_tree, X_test_tree

    def feature_selection(self, X_train_pos, X_train_dep, X_train_tree, y_train):
        """Apply feature selection using chi-square test."""
        print("Selecting top features using Chi-Square test...")
        selector_pos = SelectKBest(chi2, k=1000).fit(X_train_pos, y_train)
        selector_dep = SelectKBest(chi2, k=1000).fit(X_train_dep, y_train)
        selector_tree = SelectKBest(chi2, k=1000).fit(X_train_tree, y_train)

        X_train_pos_selected = selector_pos.transform(X_train_pos)
        X_train_dep_selected = selector_dep.transform(X_train_dep)
        X_train_tree_selected = selector_tree.transform(X_train_tree)
        
        return X_train_pos_selected, X_train_dep_selected, X_train_tree_selected

    # def train_and_evaluate_model(self, X_train_combined, y_train, X_test_combined, y_test):
    #     """Train the stacking model and evaluate accuracy."""
    #     print("Training Stacking model...")

    #     # Convert sparse matrices to dense format
    #     X_train_combined_dense = X_train_combined.toarray()
    #     X_test_combined_dense = X_test_combined.toarray()

    #     # Train the stacking model
    #     self.stacking_model.fit(X_train_combined_dense, y_train)
        
    #     print("Evaluating model...")
    #     y_pred = self.stacking_model.predict(X_test_combined_dense)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     print(f'Accuracy: {accuracy:.4f}')
    #     print("Classification Report:")
    #     print(classification_report(y_test, y_pred))
        
    #     return accuracy, y_pred

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
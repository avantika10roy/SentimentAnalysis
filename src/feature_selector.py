# Dependencies
import tqdm
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif


# Feature Selection
class TextFeatureSelector:
    """
    A class for implementing various feature selection techniques for text data
    
    Attributes:
    -----------
        X           { spmatrix } : Feature matrix
        
        y           { ndarray }  : Target labels

        feature_names { list }   : Names of features
        
        n_features    { int }    : Number of features to select
    """
    
    def __init__(self, X: spmatrix, y: np.ndarray, feature_names: list, n_features: int = None) -> None:
        """
        Initialize TextFeatureSelector with feature matrix and labels
        
        Arguments:
        ----------
            X             : Sparse feature matrix
            
            y             : Target labels
            
            feature_names : List of feature names
            
            n_features    : Number of features to select (default: 100% of input features)
            
        Raises:
        -------
            ValueError    : If inputs are invalid or incompatible
        """
        if (X.shape[0] != len(y)):
            raise ValueError("Number of samples in X and y must match")
            
        if (X.shape[1] != len(feature_names)):
            raise ValueError("Number of features must match length of feature_names")
            
        self.X             = X
        self.y             = y
        self.feature_names = feature_names
        self.n_features    = n_features or X.shape[1]  # Default 100% of the input features
        
        
    def chi_square_selection(self) -> tuple:
        """
        Perform chi-square feature selection
        
        Returns:
        --------
            { tuple } : Tuple containing: - Selected feature indices
                                          - Chi-square scores
        """
        try:
            print("Performing chi-square feature selection...")
            
            # Scale features to non-negative for chi-square
            scaler            = MinMaxScaler()
            X_scaled          = scaler.fit_transform(self.X.toarray())
            
            # Apply chi-square selection
            selector          = SelectKBest(score_func = chi2, 
                                            k          = self.n_features)
            
            selector.fit(X_scaled, self.y)
            
            # Get selected features and scores
            selected_features = np.where(selector.get_support())[0]
            scores            = selector.scores_
            
            # Sort features by importance
            sorted_idx        = np.argsort(scores)[::-1]
            selected_features = sorted_idx[:self.n_features]
            
            print(f"Selected {len(selected_features)} features using chi-square")
            
            return selected_features, scores
            
        except Exception as e:
            raise
            
    def information_gain_selection(self) -> tuple:
        """
        Perform information gain feature selection
        
        Returns:
        --------
            { tuple } : Tuple containing: - Selected feature indices
                                          - Information gain scores
        """
        try:
            print("Performing information gain selection...")
            
            # Calculate mutual information scores
            selector          = SelectKBest(score_func = mutual_info_classif, 
                                            k          = self.n_features)
            selector.fit(self.X, self.y)
            
            # Get selected features and scores
            selected_features = np.where(selector.get_support())[0]
            scores            = selector.scores_
            
            # Sort features by importance
            sorted_idx        = np.argsort(scores)[::-1]
            selected_features = sorted_idx[:self.n_features]
            
            print(f"Selected {len(selected_features)} features using information gain")
            
            return selected_features, scores
            
        except Exception as e:
            raise
            
    def correlation_based_selection(self, threshold: float = 0.8) -> np.ndarray:
        """
        Perform correlation-based feature selection
        
        Arguments:
        ----------
            threshold { float } : Correlation threshold for feature removal
            
        Returns:
        --------
               { ndarray }      :  Selected feature indices
        """
        try:
            print("Performing correlation-based selection...")
            
            # Convert sparse matrix to dense for correlation calculation
            X_dense         = self.X.toarray()
            
            # Calculate correlation matrix
            corr_matrix     = np.corrcoef(X_dense.T)
            
            # Find highly correlated feature pairs
            high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
            
            # Keep track of features to remove
            to_remove       = set()
            
            # For each pair of highly correlated features
            for i, j in zip(*high_corr_pairs):
                if ((i != j) and (i not in to_remove) and (j not in to_remove)):
                    # Calculate correlation with target for both features
                    corr_i = mutual_info_score(X_dense[:, i], self.y)
                    corr_j = mutual_info_score(X_dense[:, j], self.y)
                    
                    # Remove feature with lower correlation to target
                    if (corr_i < corr_j):
                        to_remove.add(i)
                        
                    else:
                        to_remove.add(j)
            
            # Get selected features
            all_features      = set(range(self.X.shape[1]))
            selected_features = np.array(list(all_features - to_remove))
            
            # Select top k features if more than n_features remain
            if (len(selected_features) > self.n_features):
                # Calculate mutual information for remaining features
                mi_scores         = mutual_info_classif(self.X[:, selected_features], self.y)
                top_k_idx         = np.argsort(mi_scores)[::-1][:self.n_features]
                selected_features = selected_features[top_k_idx]
            
            print(f"Selected {len(selected_features)} features using correlation-based selection")
            
            return selected_features
            
        except Exception as e:
            raise
            
    def recursive_feature_elimination(self, estimator = None, cv: int = 5) -> tuple:
        """
        Perform Recursive Feature Elimination with cross-validation
        
        Arguments:
        ----------
            estimator  : Classifier to use (default: LogisticRegression)

            cv         : Number of cross-validation folds
            
        Returns:
        --------
            { tuple }  : Tuple containing: - Selected feature indices
                                           - Feature importance rankings
        """
        try:
            print("Performing recursive feature elimination...")
            
            # Use logistic regression if no estimator provided
            if (estimator is None):
                estimator = LogisticRegression(max_iter=1000)
            
            # Perform RFE with cross-validation
            selector = RFECV(estimator              = estimator,
                             min_features_to_select = self.n_features,
                             cv                     = cv,
                             n_jobs                 = -1)
            
            selector.fit(self.X, self.y)
            
            # Get selected features and rankings
            selected_features = np.where(selector.support_)[0]
            rankings          = selector.ranking_
            
            print(f"Selected {len(selected_features)} features using RFE")
            
            return selected_features, rankings
            
        except Exception as e:
            raise
           
        
    def forward_selection(self, estimator = None, cv: int = 5) -> np.ndarray:
        """
        Perform forward feature selection
        
        Arguments:
        ----------
            estimator : Classifier to use (default: LogisticRegression)
            
            cv        : Number of cross-validation folds
            
        Returns:
        --------
            Selected feature indices
        """
        try:
            print("Performing forward selection...")
            
            if (estimator is None):
                estimator = LogisticRegression(max_iter=1000)
            
            selected_features  = list()
            remaining_features = list(range(self.X.shape[1]))
            
            for i in tqdm(range(self.n_features)):
                best_score   = -np.inf
                best_feature = None
                
                # Try adding each remaining feature
                for feature in remaining_features:
                    current_features = selected_features + [feature]
                    X_subset         = self.X[:, current_features]
                    
                    # Calculate cross-validation score
                    scores = cross_val_score(estimator, 
                                             X_subset, 
                                             self.y,
                                             cv      = cv, 
                                             scoring = 'accuracy')
                    
                    avg_score = np.mean(scores)
                    
                    if (avg_score > best_score):
                        best_score   = avg_score
                        best_feature = feature
                
                if (best_feature is not None):
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)
                
            print(f"Selected {len(selected_features)} features using forward selection")
            
            return np.array(selected_features)
            
        except Exception as e:
            raise
            
    def backward_elimination(self, estimator = None, cv: int = 5) -> np.ndarray:
        """
        Perform backward feature elimination
        
        Arguments:
        ----------
            estimator : Classifier to use (default: LogisticRegression)
            
            cv        : Number of cross-validation folds
            
        Returns:
        --------
            Selected feature indices
        """
        try:
            print("Performing backward elimination...")
            
            if (estimator is None):
                estimator = LogisticRegression(max_iter=1000)
            
            remaining_features = list(range(self.X.shape[1]))
            
            while len(remaining_features) > self.n_features:
                best_score    = -np.inf
                worst_feature = None
                
                # Try removing each feature
                for feature in remaining_features:
                    current_features = [f for f in remaining_features if f != feature]
                    X_subset         = self.X[:, current_features]
                    
                    # Calculate cross-validation score
                    scores           = cross_val_score(estimator, 
                                                       X_subset, 
                                                       self.y,
                                                       cv      = cv, 
                                                       scoring = 'accuracy')
                
                    avg_score = np.mean(scores)
                    
                    if (avg_score > best_score):
                        best_score    = avg_score
                        worst_feature = feature
                
                if (worst_feature is not None):
                    remaining_features.remove(worst_feature)
            
            print(f"Selected {len(remaining_features)} features using backward elimination")
            return np.array(remaining_features)
            
        except Exception as e:
            raise
            
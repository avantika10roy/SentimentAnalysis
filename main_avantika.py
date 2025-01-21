# Dependencies
import numpy as np
import pandas as pd
from config import DATA_PATH
from config import BATCH_SIZE
from scipy.sparse import hstack
from config import TEST_DATA_PATH
from sklearn.utils import shuffle
from src.helper import batch_generator
from src.data_loader import load_csv_data
#from config import SENTIMENT_ANALYSIS_SVM_RESULT
from src.text_preprocessor import TextPreprocessor
from src.feature_selector import TextFeatureSelector
from src.sentiment_analyzer import SentimentAnalyzer
from sklearn.model_selection import train_test_split
from config import SENTIMENT_ANALYSIS_SVM_RBF_RESULT
from config import SENTIMENT_ANALYSIS_LOGISTIC_RESULT
from config import SENTIMENT_ANALYSIS_LIGHTGBM_RESULT
from config import SENTIMENT_ANALYSIS_ADABOOST_RESULT
from src.exploratory_data_analyzer import SentimentEDA
from config import SENTIMENT_ANALYSIS_LOGISTIC_RESULT
from config import SENTIMENT_ANALYSIS_LIGHTGBM_RESULT
from config import SENTIMENT_ANALYSIS_RANDOM_FOREST_RESULT
from config import SENTIMENT_ANALYSIS_GAUSSIAN_NAIVE_BAYES_RESULT
from src.contextual_feature_engineering import Contextual_Features
from config import SENTIMENT_ANALYSIS_MULTINOMIAL_NAIVE_BAYES_RESULT
from src.word_level_feature_engineering import TextFeatureEngineering
from config import SENTIMENT_ANALYSIS_SVM_RBF_RESULT_WITH_CONTEXTUALS
from config import SENTIMENT_ANALYSIS_ADABOOST_RESULT_WITH_CONTEXTUALS
from config import SENTIMENT_ANALYSIS_LOGISTIC_GAUSSIAN_NAIVE_BAYES_RESULT
from config import SENTIMENT_ANALYSIS_GAUSSIAN_NAIVE_BAYES_RESULT_WITH_CONTEXTUALS
from config import SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT_WITH_CONTEXTUALS
from config import SENTIMENT_ANALYSIS_LOGISTIC_DECISION_TREE_RESULT_WITH_CONTEXTUALS


# Load the data
imdb_ratings_data                            = load_csv_data(filepath = DATA_PATH)

imdb_ratings_data                            = imdb_ratings_data.head(20000)

# Preprocess the text
preprocessor                                 = TextPreprocessor()
imdb_ratings_data["clean_text"]              = imdb_ratings_data["review"].apply(preprocessor.clean_text)
"""
# Initialize EDA
# eda                                          = SentimentEDA(df               = imdb_ratings_data, 
#                                                             text_column      = "clean_text", 
#                                                             sentiment_column = "sentiment", 
#                                                             output_dir       = "results/EDA_Results")
# eda.run_full_eda()
"""
# Initialize the feature engineering class
feature_eng                                  = TextFeatureEngineering(texts        = imdb_ratings_data['clean_text'].tolist(),
                                                                      max_features = 10000,
                                                                      ngram_range  = (1, 3)
                                                                     )
# Contextual Feature Engineering
contextuals = Contextual_Features(texts        = imdb_ratings_data['clean_text'].tolist(),
                                  max_features = 10000,
                                  ngram_range  = (2, 2))

"""
#feature_eng                                 = ClassFeatureEngineering(texts   = imdb_ratings_data['clean_text'].tolist(),
#                                                                      labels  = imdb_ratings_data['sentiment'].tolist()
#                                                                      ) 
"""
# Create specific feature types : Generate feature matrices
#count_vectorizer, count_features            = feature_eng.create_count_bow()
freq_vectorizer, freq_features               = feature_eng.create_frequency_bow()
#binary_vectorizer, binary_features          = feature_eng.create_binary_bow()
#tfidf_vectorizer, tfidf_features            = feature_eng.create_tfidf()
std_tfidf_vectorizer, std_tfidf_features     = feature_eng.create_standardized_tfidf()
#bm25_transformer, bm25_features             = feature_eng.create_bm25()
bm25f_transformer, bm25f_features            = feature_eng.create_bm25f()
#bm25l_transformer, bm25l_features           = feature_eng.create_bm25l()
#bm25t_transformer, bm25t_features           = feature_eng.create_bm25t()
bm25_plus_transformer, bm25_plus_features    = feature_eng.create_bm25_plus()
skipgrams_vectorizer, skipgram_features      = feature_eng.create_skipgrams()
pos_ngram_vectorizer, pos_ngram_features     = feature_eng.create_positional_ngrams()
window_vectorizer, window_features           = contextuals.window_based()
position_vectorizer, positional_features     = contextuals.position_based()
ngram_vectorizer, trigrams                   = contextuals.generate_ngrams()
cross_doc_vectorizer, tfidf_matrix           = contextuals.cross_document()

# Combine feature matrices

combined_features                            = hstack([#count_features, 
                                                       freq_features, 
                                                       #binary_features, 
                                                       #tfidf_features, 
                                                       std_tfidf_features,
                                                       #bm25_features,
                                                       bm25f_features,
                                                       #bm25l_features,
                                                       #bm25t_features,
                                                       bm25_plus_features,
                                                       skipgram_features,
                                                       pos_ngram_features,
                                                       window_features,
                                                       positional_features,
                                                       trigrams,
                                                       tfidf_matrix
                                                     ])

# Combine feature names


feature_names                                = (list(freq_vectorizer.get_feature_names_out()) +
                                                list(std_tfidf_vectorizer.get_feature_names_out()) +
                                                list(bm25f_transformer.count_vectorizer.get_feature_names_out()) +
                                                list(bm25_plus_transformer.count_vectorizer.get_feature_names_out()) +
                                                list(skipgrams_vectorizer.get_feature_names_out()) +
                                                list(pos_ngram_vectorizer.get_feature_names_out()) +
                                                list(window_vectorizer.get_feature_names_out()) +
                                                list(position_vectorizer.get_feature_names_out()) +
                                                list(ngram_vectorizer.get_feature_names_out()) +
                                                list(cross_doc_vectorizer.get_feature_names_out())
                                               )


# Feature Selection
# Initialize the feature selector
feature_selector                             = TextFeatureSelector(X             = combined_features,
                                                                   y             = imdb_ratings_data['sentiment'].values,
                                                                   feature_names = feature_names,
                                                                   n_features    = None,
                                                                  )

# Perform Feature Selection
# Chi-Square Selection
chi_square_features, chi_square_scores       = feature_selector.chi_square_selection()

# Information Gain Selection
# ig_features, ig_scores                      = feature_selector.information_gain_selection()

# Correlation-Based Selection
#corr_features                               = feature_selector.correlation_based_selection()

# Recursive Feature Elimination
#rfe_features, rfe_rankings                  = feature_selector.recursive_feature_elimination()

# Forward Selection
#forward_features                            = feature_selector.forward_selection()

# Backward Elimination
#backward_features                           = feature_selector.backward_elimination()

# Get selected features matrix
# selected_combined_features                    = combined_features[:, chi_square_features]

# Get selected features matrix
selected_combined_features                    = combined_features[:, chi_square_features]


# Sentiment Analysis
sentiment_analyzer                            = SentimentAnalyzer(X                        = selected_combined_features, 
                                                                  y                        = imdb_ratings_data["sentiment"].values,
                                                                  feature_eng              = feature_eng,
                                                                  vectorizers              = (freq_vectorizer, std_tfidf_vectorizer, bm25f_transformer, bm25_plus_transformer, skipgrams_vectorizer, pos_ngram_vectorizer, ngram_vectorizer, window_vectorizer, position_vectorizer, cross_doc_vectorizer),
                                                                  selected_feature_indices = chi_square_features)

# # Train a logistic regression model
# logistic_decision_tree_model                  = sentiment_analyzer.train_model(model_type = "logistic_decision_tree")

# # Evaluate the logistic regression model
# evaluation_results                            = sentiment_analyzer.evaluate_model(logistic_decision_tree_model)

# # Predict using the trained model
# test_data                                     = load_csv_data(filepath = TEST_DATA_PATH)

# logistic_predictions, unseen_accuracy         = sentiment_analyzer.test_on_unseen_data(model              = logistic_model, 
#                                                                                        unseen_texts       = list(test_data['Text']),
#                                                                                        unseen_labels      = list(test_data['Sentiment']),
#                                                                                        freq_features      = freq_vectorizer.transform(test_data['Text']),
#                                                                                        std_tfidf_features = std_tfidf_vectorizer.transform(test_data['Text']),
#                                                                                        bm25f_features     = bm25f_transformer.transform(test_data['Text']),
#                                                                                        bm25_plus_features = bm25_plus_transformer.transform(test_data['Text']),
#                                                                                        skipgram_features  = skipgrams_vectorizer.transform(test_data['Text']),
#                                                                                        pos_ngram_features = pos_ngram_vectorizer.transform(test_data['Text']),
#                                                                                        window_features     = window_vectorizer.transform(test_data['Text']),
#                                                                                        positional_features = position_vectorizer.transform(test_data['Text']),
#                                                                                        ngram_features      = ngram_vectorizer.transform(test_data['Text']),
#                                                                                        cross_doc_features  = cross_doc_vectorizer.transform(test_data['Text'])
#                                                                                       )   

# all_test_data                                 = {'texts'            : list(test_data['Text']), 
#                                                  'true_labels'      : list(test_data['Sentiment']), 
#                                                  'predicted_labels' : list(logistic_predictions)
#                                                 }

# logistic_prediction_df                             = pd.DataFrame.from_dict(data   = all_test_data, 
#                                                                        orient = 'index').T

# logistic_prediction_df.to_csv(path_or_buf = SENTIMENT_ANALYSIS_LOGISTIC_RESULT,
#                               index       = False)

# print (f"Sentiment Analysis result by Losgistic Regression Model has been saved to : {SENTIMENT_ANALYSIS_LOGISTIC_RESULT}")                                                                    


# Train an Adaboost model
mlp_model                                       = sentiment_analyzer.train_model(model_type="multilayer_perceptron")

# Evaluate the SVM model
evaluate_mlp_results                            = sentiment_analyzer.evaluate_model(mlp_model)

# Predict using the trained model
test_data                                       = load_csv_data(filepath = TEST_DATA_PATH)

mlp_predictions, unseen_accuracy                = sentiment_analyzer.test_on_unseen_data(model               = mlp_model,
                                                                                         unseen_texts        = list(test_data['Text']),
                                                                                         unseen_labels       = list(test_data['Sentiment']),
                                                                                         freq_features       = freq_vectorizer.transform(test_data['Text']),
                                                                                         std_tfidf_features  = std_tfidf_vectorizer.transform(test_data['Text']),
                                                                                         bm25f_features      = bm25f_transformer.transform(test_data['Text']),
                                                                                         bm25_plus_features  = bm25_plus_transformer.transform(test_data['Text']),
                                                                                         skipgram_features   = skipgrams_vectorizer.transform(test_data['Text']),
                                                                                         pos_ngram_features  = pos_ngram_vectorizer.transform(test_data['Text']),
                                                                                         window_features     = window_vectorizer.transform(test_data['Text']),
                                                                                         positional_features = position_vectorizer.transform(test_data['Text']),
                                                                                         ngram_features      = ngram_vectorizer.transform(test_data['Text']),
                                                                                         cross_doc_features  = cross_doc_vectorizer.transform(test_data['Text']) 
                                                                                         )

all_test_data                                 = {'texts'            : list(test_data['Text']), 
                                                 'true_labels'      : list(test_data['Sentiment']), 
                                                 'predicted_labels' : list(mlp_predictions)
                                                }

mlp_prediction_df                             = pd.DataFrame.from_dict(data   = all_test_data, 
                                                                       orient = 'index').T

mlp_prediction_df.to_csv(path_or_buf = SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT_WITH_CONTEXTUALS,
                          index       = False)

print (f"Sentiment Analysis result by Multilayer Perceptron Model has been saved to : {SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT_WITH_CONTEXTUALS}")


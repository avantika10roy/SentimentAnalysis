## ----- DONE BY PRIYAM PAL -----


# DEPENDENCIES

import numpy as np
import pandas as pd

from config import DATA_PATH
from config import BATCH_SIZE
from config import MODEL_NAME
from config import KERNEL_NAME
from config import MAX_FEATURES
from config import ELMO_MODEL_URL
from config import TEST_DATA_PATH
from config import GLOVE_MODEL_PATH
from config import SENTIMENT_ANALYSIS_SVM_RBF_BY_SEMANTIC_FEAT_RESULT

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from src.helper import batch_generator
from src.data_loader import load_csv_data
from src.text_preprocessor import TextPreprocessor
from src.feature_selector import TextFeatureSelector
from src.sentiment_analyzer import SentimentAnalyzer
from src.transform_vectorizer import vector_transform
from src.exploratory_data_analyzer import SentimentEDA
from src.semantic_Feature_Engineering import Semantic_Feature_Engineering

import warnings
warnings.filterwarnings(action = 'ignore')

# LOAD THE DATA
imdb_ratings_data                            = load_csv_data(filepath = DATA_PATH)

# PREPROCESSING THE DATA
preprocessor                                 = TextPreprocessor()
imdb_ratings_data["clean_text"]              = imdb_ratings_data["review"].apply(preprocessor.clean_text)


# ----- EXPLORATORY DATA ANALYSIS -----

# # INTIALISING THE EDA CLASS
# eda                                          = SentimentEDA(df               = imdb_ratings_data, 
#                                                             text_column      = "clean_text", 
#                                                             sentiment_column = "sentiment", 
#                                                             output_dir       = "results/EDA_Results")
# eda.run_full_eda()



# ------------------------- PRIYAM -------------------------

# --------------- SEMANTIC FEATURE ENGINEERING ---------------


# INITIALISING THE SEMANTIC FEATURE ENGINEERING CLASS
semantic_Feature_Eng                    = Semantic_Feature_Engineering(texts        = imdb_ratings_data['clean_text'].tolist(), 
                                                                       max_features = MAX_FEATURES
                                                                       )
# INITIALISING THE CONTEXTUAL EMBEDDING CLASS
contextual_Embedding                    = semantic_Feature_Eng.Contextual_Embedding(texts = imdb_ratings_data['clean_text'].tolist())



# -----  CREATING THE FEATURES -----

w2v_model, w2v_features                 = semantic_Feature_Eng.word2vec_cbow()
# glove_embeddings, glove_model           = semantic_Feature_Eng.glove(GLOVE_MODEL_PATH)
# fasttext_model, fasttext_features       = semantic_Feature_Eng.fasttext()
# wordnet_model, wordnet_features         = semantic_Feature_Eng.wordnet()

# CONVERTING THE FEATURES INTO FEATURE MATRIX
w2v_sparse                              = csr_matrix(w2v_features)
# glove_sparse                            = csr_matrix(glove_embeddings)
# fasttext_sparse                         = csr_matrix(fasttext_features)

# COMBINING THE FEATURES
combined_features                       = hstack([w2v_sparse, 
                                                #   glove_sparse, 
                                                #  fasttext_sparse
                                                  ])

print(f"Combined Feature Matrix Shape: {combined_features.shape}")


# ----- EXTRACTING THE FEATURE NAMES -----

feature_names                            = []

w2v_feature_names                        = w2v_model.wv.index_to_key[:MAX_FEATURES]
# fasttext_feature_names                   = fasttext_model.wv.index_to_key[:MAX_FEATURES]

feature_names                            = list(w2v_feature_names)

print(f"Number of feature names extracted: {len(feature_names)}")


# ----- SELECTING THE FEATURES -----

# FEATURE SELECTOR
feature_selector                         = TextFeatureSelector(X             = combined_features,
                                                               y             = imdb_ratings_data['sentiment'].values,
                                                               feature_names = feature_names,
                                                               n_features    = MAX_FEATURES
                                                               )

# CHI-SQUARE SELECTION
chi_square_features, chi_square_scores   = feature_selector.chi_square_selection()

# COMBINING THE FEATURES
selected_combined_features               = combined_features[:, chi_square_features]



# ----- SENTIMENTAL ANALYSIS -----

sentiment_analyzer                       = SentimentAnalyzer(X                        = selected_combined_features, 
                                                             y                        = imdb_ratings_data["sentiment"].values,
                                                             feature_eng              = semantic_Feature_Eng,
                                                             vectorizers              = None,
                                                             selected_feature_indices = chi_square_features
                                                             )


# ----- MODEL FITTING ON TRAINING DATA -----


# TRAIN THE MODEL
trained_model                             = sentiment_analyzer.train_model(model_type = MODEL_NAME, kernel = KERNEL_NAME)

# EVALUATING THE RESULTS OF THE MODEL
evaluation_results                        = sentiment_analyzer.evaluate_model(trained_model)


# ----- TRANSFORMING THE VECTORIZERS -----

test_data                                 = load_csv_data(filepath = TEST_DATA_PATH)

# TRANSFORMING THE VECTORS USING PRETRAINED MODEL OF WORD2VEC AND FASTTEXT
w2v_features_transformed                  = vector_transform(list(test_data['Text']), w2v_model)
# fasttext_features_transformed             = vector_transform(list(test_data['Text']), fasttext_model)

# COMBINING THE FEATURES
combined_features_transformed             = np.hstack([w2v_features_transformed, 
                                                       # fasttext_features_transformed
                                                      ])

# CONVERTING TO SPARSE MATRIX
combined_features_sparse                  = csr_matrix(combined_features_transformed)


# # ----- PREDICT THE TRAINED MODEL USING UNSEEN DATA -----

model_predictions, unseen_accuracy        = sentiment_analyzer.test_on_unseen_data(model              = trained_model, 
                                                                                   unseen_texts       = list(test_data['Text']),
                                                                                   unseen_labels      = list(test_data['Sentiment']),
                                                                                   combined_features  = combined_features_sparse
                                                                                   )

print(f"Accuracy on unseen data: {unseen_accuracy:.4f}")


all_test_data                              = {'texts'            : list(test_data['Text']), 
                                              'true_labels'      : list(test_data['Sentiment']), 
                                              'predicted_labels' : list(model_predictions)
                                              }

model_prediction_df                        = pd.DataFrame.from_dict(data   = all_test_data, 
                                                                    orient = 'index').T

model_prediction_df.to_csv(path_or_buf     = SENTIMENT_ANALYSIS_SVM_RBF_BY_SEMANTIC_FEAT_RESULT,
                           index           = False)

print (f"Sentiment Analysis result by {MODEL_NAME} Model of Max Features {MAX_FEATURES} has been saved to : {SENTIMENT_ANALYSIS_SVM_RBF_BY_SEMANTIC_FEAT_RESULT}")

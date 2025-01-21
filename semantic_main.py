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
from config import SENTIMENT_ANALYSIS_LABEL_PROP_BY_ALL_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_GAUSSIAN_NB_BY_ALL_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_SVM_RBF_BY_SEMANTIC_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_LOGISTIC_REG_BY_ALL_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_LIGHT_GBM_BY_SEMANTIC_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_LABEL_PROP_BY_SEMANTIC_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_SVM_SIGMOID_BY_SEMANTIC_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_GAUSSIAN_NB_BY_SEMANTIC_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_LOGISTIC_REG_BY_SEMANTIC_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_RANDOM_FOREST_BY_SEMANTIC_FEAT_RESULT
from config import SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_BY_ALL_FEAT_RESULT

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
from src.contextual_feature_engineering import Contextual_Features
from src.word_level_feature_engineering import TextFeatureEngineering
from src.semantic_feature_engineering import Semantic_Feature_Engineering

import warnings
warnings.filterwarnings(action = 'ignore')

SAVE_PATH_VARIABLE                           = SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_BY_ALL_FEAT_RESULT

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

# INITIALISING THE CONTEXTUAL EMBEDDING CLASS INSIDE SEMANTIC FEATURE ENGINEERING CLASS
contextual_Embedding                    = semantic_Feature_Eng.Contextual_Embedding(texts = imdb_ratings_data['clean_text'].tolist())

# INITIALISING THE WORD - LEVEL FEATURE ENGINEERING CLASS
word_level_feature_eng                  = TextFeatureEngineering(texts        = imdb_ratings_data['clean_text'].tolist(), 
                                                                 max_features = MAX_FEATURES, 
                                                                 ngram_range  = (1, 3)
                                                                 )
# INITIALISING THE CONTEXTUAL FEATURE ENGINEERING CLASS
contextuals                             = Contextual_Features(texts           = imdb_ratings_data['clean_text'].tolist(),
                                                              max_features    = MAX_FEATURES,
                                                              ngram_range     = (2, 2))


# ----------  CREATING THE FEATURES ----------


# ----- WORD - LEVEL FEATURES -----

# count_vectorizer, count_features             = word_level_feature_eng.create_count_bow()
freq_vectorizer, freq_features               = word_level_feature_eng.create_frequency_bow()
# binary_vectorizer, binary_features           = word_level_feature_eng.create_binary_bow()
# tfidf_vectorizer, tfidf_features             = word_level_feature_eng.create_tfidf()
std_tfidf_vectorizer, std_tfidf_features     = word_level_feature_eng.create_standardized_tfidf()
# bm25_transformer, bm25_features              = word_level_feature_eng.create_bm25()
bm25f_transformer, bm25f_features            = word_level_feature_eng.create_bm25f()
# bm25l_transformer, bm25l_features            = word_level_feature_eng.create_bm25l()
# bm25t_transformer, bm25t_features            = word_level_feature_eng.create_bm25t()
bm25_plus_transformer, bm25_plus_features    = word_level_feature_eng.create_bm25_plus()
skipgrams_vectorizer, skipgram_features      = word_level_feature_eng.create_skipgrams()
pos_ngram_vectorizer, pos_ngram_features     = word_level_feature_eng.create_positional_ngrams()

# ----- CONTEXTUALS FEATURES -----

window_vectorizer, window_features           = contextuals.window_based()
# position_vectorizer, positional_features     = contextuals.position_based()
ngram_vectorizer, trigrams                   = contextuals.generate_ngrams()
# cross_doc_vectorizer, tfidf_matrix           = contextuals.cross_document()


# ----- SEMANTIC FEATURES -----

w2v_model, w2v_features                 = semantic_Feature_Eng.word2vec_cbow()
# glove_embeddings, glove_model           = semantic_Feature_Eng.glove(GLOVE_MODEL_PATH)
# fasttext_model, fasttext_features       = semantic_Feature_Eng.fasttext()
# wordnet_model, wordnet_features         = semantic_Feature_Eng.wordnet()


# CONVERTING THE FEATURES INTO FEATURE MATRIX
w2v_sparse                              = csr_matrix(w2v_features)
# glove_sparse                            = csr_matrix(glove_embeddings)
# fasttext_sparse                         = csr_matrix(fasttext_features)

# COMBINING THE SEMANTIC, WORD - LEVEL FEATURES, CONTEXTUAL FEATURES
combined_features                       = hstack([w2v_sparse, 
                                                  # glove_sparse, 
                                                  # fasttext_sparse,
                                                  freq_features, 
                                                  std_tfidf_features,
                                                  bm25f_features,
                                                  bm25_plus_features,
                                                  skipgram_features,
                                                  pos_ngram_features,
                                                  window_features,
                                                  # positional_features,
                                                  trigrams,
                                                  # tfidf_matrix 
                                                  ])

print(f"Combined Feature Matrix Shape: {combined_features.shape}")


# ----- EXTRACTING THE FEATURE NAMES -----

feature_names                            = []

w2v_feature_names                        = w2v_model.wv.index_to_key[:MAX_FEATURES]
# fasttext_feature_names                   = fasttext_model.wv.index_to_key[:MAX_FEATURES]

# COMBINING THE FEATURE NAMES OF SEMANTIC, WORD-LEVEL, CONTEXTUAL FEATURES
feature_names                            = (list(w2v_feature_names) + 
                                            # list(fasttext_feature_names) + 
                                            list(freq_vectorizer.get_feature_names_out()) +
                                            list(std_tfidf_vectorizer.get_feature_names_out()) +
                                            list(bm25f_transformer.count_vectorizer.get_feature_names_out()) +
                                            list(bm25_plus_transformer.count_vectorizer.get_feature_names_out()) +
                                            list(skipgrams_vectorizer.get_feature_names_out()) +
                                            list(pos_ngram_vectorizer.get_feature_names_out()) + 
                                            list(window_vectorizer.get_feature_names_out()) +
                                            # list(position_vectorizer.get_feature_names_out()) +
                                            # list(cross_doc_vectorizer.get_feature_names_out())
                                            list(ngram_vectorizer.get_feature_names_out())
                                           )

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


# VECTORIZERS TUPLE
vectorizers_tuple                        = (w2v_model,
                                            # fasttext_model,
                                            freq_vectorizer,
                                            std_tfidf_vectorizer, 
                                            bm25f_transformer,
                                            bm25_plus_features, 
                                            skipgrams_vectorizer, 
                                            pos_ngram_vectorizer,
                                            window_vectorizer,
                                            # position_vectorizer,
                                            ngram_vectorizer,
                                            # cross_doc_vectorizer
                                            )

# ----- SENTIMENTAL ANALYSIS -----

sentiment_analyzer                       = SentimentAnalyzer(X                        = selected_combined_features, 
                                                             y                        = imdb_ratings_data["sentiment"].values,
                                                             feature_eng              = semantic_Feature_Eng,
                                                             vectorizers              = vectorizers_tuple,
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
combined_features_transformed             = np.hstack([w2v_features_transformed])

# CONVERTING TO SPARSE MATRIX
combined_features_sparse                  = csr_matrix(combined_features_transformed)


# # ----- PREDICT THE TRAINED MODEL USING UNSEEN DATA USING SEMANTIC, WORD-LEVEL, CONTEXTUAL FEATURES -----

model_predictions, unseen_accuracy        = sentiment_analyzer.test_on_unseen_data(model               = trained_model, 
                                                                                   unseen_texts        = list(test_data['Text']),
                                                                                   unseen_labels       = list(test_data['Sentiment']),
                                                                                   combined_features   = combined_features_sparse,
                                                                                   freq_features       = freq_vectorizer.transform(test_data['Text']),
                                                                                   std_tfidf_features  = std_tfidf_vectorizer.transform(test_data['Text']),
                                                                                   bm25f_features      = bm25f_transformer.transform(test_data['Text']),
                                                                                   bm25_plus_features  = bm25_plus_transformer.transform(test_data['Text']),
                                                                                   skipgram_features   = skipgrams_vectorizer.transform(test_data['Text']),
                                                                                   pos_ngram_features  = pos_ngram_vectorizer.transform(test_data['Text']),
                                                                                   window_features     = window_vectorizer.transform(test_data['Text']),
                                                                                   # positional_features = position_vectorizer.transform(test_data['Text']),
                                                                                   ngram_features      = ngram_vectorizer.transform(test_data['Text']),
                                                                                   # cross_doc_features  = cross_doc_vectorizer.transform(test_data['Text']) 
                                                                                   )


all_test_data                              = {'texts'            : list(test_data['Text']), 
                                              'true_labels'      : list(test_data['Sentiment']), 
                                              'predicted_labels' : list(model_predictions)
                                              }

model_prediction_df                        = pd.DataFrame.from_dict(data   = all_test_data, 
                                                                    orient = 'index').T

model_prediction_df.to_csv(path_or_buf     = SAVE_PATH_VARIABLE,
                           index           = False)

print (f"Sentiment Analysis result by {MODEL_NAME} Model of Max Features {MAX_FEATURES} has been saved to : {SAVE_PATH_VARIABLE}")

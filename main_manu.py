# -- Dependencies --
import os
from tqdm import tqdm
import gensim
import numpy as np
import pandas as pd
import gensim.downloader
from scipy.sparse import hstack
from config import WORD2VEC_MODEL
from src.data_loader import load_csv_data
from config import DATA_PATH, TEST_DATA_PATH
from src.text_preprocessor import TextPreprocessor
from src.feature_selector import TextFeatureSelector
from src.exploratory_data_analyzer import SentimentEDA
from src.sentiment_analyzer_v2 import SentimentAnalyzer
from src.customized_feature_engineering import TextVectorizer
from config import SENTIMENT_ANALYSIS_LOGISTIC_WITH_CUSTOM_FEAT
from src.word_level_feature_engineering import TextFeatureEngineering

# -- Variables --
model_path                                   = f'models/{WORD2VEC_MODEL}'

# -- Load the data -- 
imdb_ratings_df                              = load_csv_data(filepath = DATA_PATH)
test_data                                    = load_csv_data(filepath = TEST_DATA_PATH)

# -- Preprocess the text -- 
tqdm.pandas()
preprocessor                                 = TextPreprocessor()
imdb_ratings_df["clean_text"]                = imdb_ratings_df["review"].progress_apply(preprocessor.clean_text)
test_data['clean_text']                      = test_data['Text'].progress_apply(preprocessor.clean_text)

# -- Initialize EDA --
'''eda                                          = SentimentEDA(df            = imdb_ratings_data, 
                                                            text_column      = "clean_text", 
                                                            sentiment_column = "sentiment", 
                                                            output_dir       = "results/EDA_Results")
#eda.run_full_eda()
'''

# -- Feature Engineering & Selection --
word_level_features                          = TextFeatureEngineering(list(imdb_ratings_df['clean_text']),
                                                                      max_features=20000)


freq_vectorizer, freq_features               = word_level_features.create_frequency_bow()
std_tfidf_vectorizer, std_tfidf_features     = word_level_features.create_standardized_tfidf()
bm25f_transformer, bm25f_features            = word_level_features.create_bm25f()
skipgrams_vectorizer, skipgram_features      = word_level_features.create_skipgrams()
pos_ngram_vectorizer, pos_ngram_features     = word_level_features.create_positional_ngrams()

combined_features                           = hstack([freq_features,
                                                      std_tfidf_features,
                                                      bm25f_features,
                                                      skipgram_features,
                                                      pos_ngram_features
                                                    ])

feature_names                                = list(freq_vectorizer.get_feature_names_out()) +list(std_tfidf_vectorizer.get_feature_names_out()) +list(bm25f_transformer.count_vectorizer.get_feature_names_out()) +list(skipgrams_vectorizer.get_feature_names_out()) +list(pos_ngram_vectorizer.get_feature_names_out())
                                        

new_selector                                 = TextFeatureSelector(X             = (combined_features),
                                                                   y             = imdb_ratings_df['sentiment'].values,
                                                                   feature_names = (feature_names),
                                                                   n_features    = 10000,
                                                                  )

chi2_features, scores                        = new_selector.chi_square_selection()
selected_features                            = list(np.array(feature_names)[chi2_features])

if not os.path.exists(model_path):
    model                                    = gensim.downloader.load(WORD2VEC_MODEL)
    model.save(model_path)


word_vec                                     = TextVectorizer(feature_names   = list(selected_features),
                                                              weight_factor   = 2.0,
                                                              use_pretrained  = True,
                                                              pretrained_path = model_path)

word_vec.load_pretrained_model()

X_train                                      = word_vec.transform(list(imdb_ratings_df['clean_text']))
y_train                                      = imdb_ratings_df['sentiment'].values

# -- Model Training -- 

sentiment_analyzer                           = SentimentAnalyzer(X_train, y_train)
model                                        = sentiment_analyzer.train_model(model_type='logistic_regression')

# -- Evaluate Model --
evaluation_results                           = sentiment_analyzer.evaluate_model(model)

unseen_test_res                              = sentiment_analyzer.test_on_unseen_data(model, list(test_data['clean_text']), word_vec.transform, list(test_data['Sentiment']))

all_test_data                                = {'texts'            : list(test_data['Text']), 
                                                'true_labels'      : list(test_data['Sentiment']), 
                                                'predicted_labels' : list(unseen_test_res['predictions'])
                                               }

prediction_df                                = pd.DataFrame.from_dict(data   = all_test_data, 
                                                                      orient = 'index').T

prediction_df.to_csv(path_or_buf     = SENTIMENT_ANALYSIS_LOGISTIC_WITH_CUSTOM_FEAT,
                     index           = False)

print (f"Sentiment Analysis result by logistic regression Model has been saved to : {SENTIMENT_ANALYSIS_LOGISTIC_WITH_CUSTOM_FEAT}")

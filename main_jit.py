import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_csv_data
from config import DATA_PATH, TEST_DATA_PATH
from src.text_preprocessor import TextPreprocessor
from src.feature_selector import TextFeatureSelector
from src.sentiment_analyzer import SentimentAnalyzer
from sklearn.model_selection import train_test_split
from config import SENTIMENT_ANALYSIS_LOGISTIC_RESULT_BY_STAT_FEAT
from config import SENTIMENT_ANALYSIS_LIGHTGBM_RESULT_BY_STAT_FEAT
from src.exploratory_data_analyzer import SentimentEDA
from src.statistical_feature_engineering import Statistical_Feature_Engineering
import warnings
warnings.filterwarnings("ignore")

# Load the data
train_data                                   = load_csv_data(filepath = DATA_PATH)
test_data                                    = load_csv_data(filepath = TEST_DATA_PATH)

test_data                                    = test_data.rename(columns={'Text':'review','Sentiment':'sentiment'})
# Preprocess the text
preprocessor                                 = TextPreprocessor()
train_data["cleaned_review"]                 = train_data["review"].apply(preprocessor.clean_text)
test_data['cleaned_review']                  = test_data['review'].apply(preprocessor.clean_text)

# Initialize EDA
'''eda                                          = SentimentEDA(df            = imdb_ratings_data, 
                                                            text_column      = "clean_text", 
                                                            sentiment_column = "sentiment", 
                                                            output_dir       = "results/EDA_Results")
#eda.run_full_eda()
'''


# Statistical Feature Engineering
feature_engineer                             = Statistical_Feature_Engineering(max_features=10000)

doc_vect, train_doc_stat = feature_engineer.create_document_statistics(text=train_data['review'].tolist(),
                                                    cleaned_text=train_data['cleaned_review'].tolist())
read_vect, train_readability = feature_engineer.create_readability_score(cleaned_text=train_data['cleaned_review'].tolist(),score='ALL')


test_doc_stat = doc_vect.transform(text=test_data['review'].tolist(), cleaned_text=test_data['cleaned_review'].tolist())
test_readability = read_vect.transform(text=test_data['cleaned_review'].tolist(), score='ALL')

# Document Statistics & Readability Scores
train_stat_features_sparse                   = hstack([train_doc_stat, train_readability])
test_stat_features_sparse                    = hstack([test_doc_stat, test_readability])

# Frequency Distribution
bow_vect, train_bow_sparse                   = feature_engineer.create_frequency_distribution(train_data['cleaned_review'].tolist())
test_bow_sparse                              = bow_vect.transform(test_data['cleaned_review'].tolist())

# Model Training
X_train_combined                             = hstack([train_stat_features_sparse, train_bow_sparse])

# Label Encoding
label_encoder                                = LabelEncoder()
y_train                                      = label_encoder.fit_transform(train_data['sentiment'])

#Feature Selection
selector                                     = TextFeatureSelector(X_train_combined, y_train, 
                                                                   feature_names=(list(bow_vect.get_feature_names_out()) +
                                                                    ['char_count', 'word_count', 'sent_count', 'AWL', 'ASL', 'UWR', 'FRE', 'GFI', 'SMOG']),
                                                                    n_features=9000)

selected_features, chi2_scores               = selector.chi_square_selection()
X_train_selected                             = X_train_combined[:, selected_features]

# Model Training
sentiment_analyzer                           = SentimentAnalyzer(X_train_selected, y_train, feature_engineer, selected_feature_indices=selected_features)
model                                        = sentiment_analyzer.train_model(model_type="logistic_regression")

# Model Evaluation
metrics                                      = sentiment_analyzer.evaluate_model(model)

# Label Encoding test data
label_encoder                                = LabelEncoder()
test_data['sentiment']                       = label_encoder.fit_transform(test_data['sentiment'])

# test_on_unseen_data
predictions, unseen_accuracy                        = sentiment_analyzer.test_on_unseen_data(model,
                                                                                        unseen_texts=test_data['review'].tolist(),
                                                                                        unseen_labels=test_data['sentiment'].values,
                                                                                        statistical_features=test_stat_features_sparse,
                                                                                        bow_feature=test_bow_sparse)

all_test_data                                 = {'texts'            : list(test_data['review']), 
                                                 'true_labels'      : list(test_data['sentiment']), 
                                                 'predicted_labels' : list(predictions)
                                                }

logistic_prediction_df                        = pd.DataFrame.from_dict(data   = all_test_data, 
                                                                       orient = 'index').T

logistic_prediction_df.to_csv(path_or_buf     = SENTIMENT_ANALYSIS_LOGISTIC_RESULT_BY_STAT_FEAT,
                              index           = False)

print (f"Sentiment Analysis result by logistic regression Model has been saved to : {SENTIMENT_ANALYSIS_LOGISTIC_RESULT_BY_STAT_FEAT}")


# DATA PATHS
DATA_PATH                                                       = 'data/IMDB_Dataset.csv'
TEST_DATA_PATH                                                  = 'data/test_data.csv'
Emotion_path                                                    = 'data/emotion_lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'


# CONFIGURATION VARIABLES
BATCH_SIZE                                                      = 250
MAX_FEATURES                                                    = 500
MODEL_NAME                                                      = "multilayer_perceptron"
KERNEL_NAME                                                     = None # IF MODEL DOESN'T HAVE KERNEL, MAKE IT NONE
GLOVE_MODEL_PATH                                                = "models/glove.6B.100d.txt"
ELMO_MODEL_URL                                                  = "https://tfhub.dev/google/elmo/3"
WORD2VEC_MODEL                                                  = "word2vec-google-news-300"
BERT_CONFIG                                                     = "models/BERT/config.json"
BERT_MODEL_SAFETENSORS                                          = "models/BERT/model.safetensors"
BERT_TOKENIZER_CONFIG                                           = "models/BERT/tokenizer_config.json"
BERT_TOKENIZER                                                  = "models/BERT/tokenizer.json"
BERT_VOCABULARY                                                 = "models/BERT/vocab.txt"
DISTILBERT_CONFIG                                               = "models/DISTILBERT/config.json"
DISTILBERT_MODEL_SAFETENSORS                                    = "models/DISTILBERT/model.safetensors"
DISTILBERT_TOKENIZER_CONFIG                                     = "models/DISTILBERT/tokenizer_config.json"
DISTILBERT_TOKENIZER                                            = "models/DISTILBERT/tokenizer.json"
DISTILBERT_VOCABULARY                                           = "models/DISTILBERT/vocab.txt"
DISTILBERT_PYTORCH_BIN                                          = "models/DISTILBERT/pytorch_model.bin"


# PARAMETER DICTIONARY
MODEL_PARAMS_DICT                                               = {'C'                 : 1.0,
                                                                   'tol'               : 0.001,
                                                                   'loss'              : 'log_loss',
                                                                   'solver'            : 'lbfgs',
                                                                   'penalty'           : 'l2',
                                                                   'max_iter'          : 1000,
                                                                   'max_depth'         : 50,
                                                                   'n_neighbors'       : 2,
                                                                   'max_features'      : 1,
                                                                   'learning_rate'     : 0.01,
                                                                   'min_samples_leaf'  : 5,
                                                                   'hidden_layer_size' : 1000,
                                                                   'l2_regularization' : 0.01,
                                                                   'min_samples_split' : 10             
                                                                  }


# RESULT PATHS
SENTIMENT_ANALYSIS_SVM_RBF_RESULT                               = 'results/sentiment_analysis_result_svm_rbf.csv'
SENTIMENT_ANALYSIS_LOGISTIC_RESULT                              = 'results/sentiment_analysis_result_logistic.csv'
SENTIMENT_ANALYSIS_LIGHTGBM_RESULT                              = 'results/sentiment_analysis_result_lightgbm.csv'
SENTIMENT_ANALYSIS_ADABOOST_RESULT                              = 'results/sentiment_analysis_result_adaboost.csv'
SENTIMENT_ANALYSIS_SVM_SIGMOID_RESULT                           = 'results/sentiment_analysis_result_svm_sigmoid.csv'
SENTIMENT_ANALYSIS_RANDOM_FOREST_RESULT                         = 'results/sentiment_analysis_result_random_forest.csv'
SENTIMENT_ANALYSIS_SVM_POLYNOMIAL_RESULT                        = 'results/sentiment_analysis_result_svm_polynomial.csv'
SENTIMENT_ANALYSIS_GRADIENT_BOOST_RESULT                        = 'results/sentiment_analysis_result_gradient_boost.csv'
SENTIMENT_ANALYSIS_LABEL_PROPAGATION_RESULT                     = 'results/sentiment_analysis_result_label_propagation.csv'
SENTIMENT_ANALYSIS_LOGISTIC_WITH_CUSTOM_FEAT                    = 'results/sentiment_analysis_result_logistic_with_coustom_features.csv'
SENTIMENT_ANALYSIS_GAUSSIAN_NAIVE_BAYES_RESULT                  = 'results/sentiment_analysis_result_gaussian_naive_bayes.csv'
SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT                 = 'results/sentiment_analysis_result_multilayer_perceptron.csv'
SENTIMENT_ANALYSIS_LOGISTIC_RESULT_BY_STAT_FEAT                 = 'results/sentiment_analysis_result_logistic_by_statistical_features.csv'
SENTIMENT_ANALYSIS_LIGHTGBM_RESULT_BY_STAT_FEAT                 = 'results/sentiment_analysis_result_lightgbm_by_statistical_features.csv'
SENTIMENT_ANALYSIS_LOGISTIC_DECISION_TREE_RESULT                = 'results/sentiment_analysis_result_logistic_model_tree.csv'
SENTIMENT_ANALYSIS_MULTINOMIAL_NAIVE_BAYES_RESULT               = 'results/sentiment_analysis_result_naive_bayes.csv'
SENTIMENT_ANALYSIS_SVM_RBF_RESULT_WITH_CONTEXTUALS              = 'results/sentiment_analysis_result_svm_rbf_with_contextuals.csv'
SENTIMENT_ANALYSIS_SVM_RBF_RESULT_WITH_CONTEXTUALS              = 'results/sentiment_analysis_result_svm_rbf_with_contextuals.csv'
SENTIMENT_ANALYSIS_SVM_RBF_BY_SEMANTIC_FEAT_RESULT              = 'results/sentiment_analysis_result_svm_rbf_by_semantic_features.csv'
SENTIMENT_ANALYSIS_ADABOOST_RESULT_WITH_CONTEXTUALS             = 'results/sentiment_analysis_result_adaboost_with_contextuals.csv'
SENTIMENT_ANALYSIS_LOGISTIC_GAUSSIAN_NAIVE_BAYES_RESULT         = 'results/sentiment_analysis_result_logistic_gaussian_naive_bayes.csv'
SENTIMENT_ANALYSIS_HIST_GRADIENT_BOOSTING_CLASSIFIER_RESULT     = 'results/sentiment_analysis_result_hist_gradient_boosting_classifier.csv'
SENTIMENT_ANALYSIS_GAUSSIAN_NAIVE_BAYES_RESULT_WITH_CONTEXTUALS = 'results/sentiment_analysis_gaussian_naive_bayes_with_contextuals.csv'
SENTIMENT_ANALYSIS_LOGISTIC_REG_BY_SEMANTIC_FEAT_RESULT         = 'results/sentiment_analysis_result_logistic_reg_by_semantic_features.csv'
SENTIMENT_ANALYSIS_SVM_RBF_BY_SEMANTIC_FEAT_RESULT              = 'results/sentiment_analysis_result_svm_rbf_by_semantic_features.csv'
SENTIMENT_ANALYSIS_SVM_SIGMOID_BY_SEMANTIC_FEAT_RESULT          = 'results/sentiment_analysis_result_svm_sigmoid_by_semantic_features.csv'
SENTIMENT_ANALYSIS_GAUSSIAN_NB_BY_SEMANTIC_FEAT_RESULT          = 'results/sentiment_analysis_result_gaussian_nb_by_semantic_features.csv'
SENTIMENT_ANALYSIS_LIGHT_GBM_BY_SEMANTIC_FEAT_RESULT            = 'results/sentiment_analysis_result_light_gbm_by_semantic_features.csv'
SENTIMENT_ANALYSIS_RANDOM_FOREST_BY_SEMANTIC_FEAT_RESULT        = 'results/sentiment_analysis_result_random_forest_by_semantic_features.csv'
SENTIMENT_ANALYSIS_LABEL_PROP_BY_SEMANTIC_FEAT_RESULT           = 'results/sentiment_analysis_result_label_prop_by_semantic_features.csv'
SENTIMENT_ANALYSIS_LOGISTIC_REG_BY_ALL_FEAT_RESULT              = 'results/sentiment_analysis_result_logistic_reg_by_all_features.csv'
SENTIMENT_ANALYSIS_GAUSSIAN_NB_BY_ALL_FEAT_RESULT               = 'results/sentiment_analysis_result_gaussian_nb_by_all_features.csv'
SENTIMENT_ANALYSIS_LABEL_PROP_BY_ALL_FEAT_RESULT                = 'results/sentiment_analysis_result_label_prop_by_all_features.csv'
SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_BY_ALL_FEAT_RESULT     = 'results/sentiment_analysis_result_multilayer_perceptron_by_all_features.csv'
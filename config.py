# DATA PATHS
DATA_PATH                                                   = 'data/IMDB_Dataset.csv'
TEST_DATA_PATH                                              = 'data/test_data.csv'

# RESULT PATHS
SENTIMENT_ANALYSIS_SVM_RBF_RESULT                           = 'results/sentiment_analysis_result_svm_rbf.csv'
SENTIMENT_ANALYSIS_LOGISTIC_RESULT                          = 'results/sentiment_analysis_result_logistic.csv'
SENTIMENT_ANALYSIS_LIGHTGBM_RESULT                          = 'results/sentiment_analysis_result_lightgbm.csv'
SENTIMENT_ANALYSIS_ADABOOST_RESULT                          = 'results/sentiment_analysis_result_adaboost.csv'
SENTIMENT_ANALYSIS_SVM_SIGMOID_RESULT                       = 'results/sentiment_analysis_result_svm_sigmoid.csv'
SENTIMENT_ANALYSIS_RANDOM_FOREST_RESULT                     = 'results/sentiment_analysis_result_random_forest.csv'
SENTIMENT_ANALYSIS_SVM_POLYNOMIAL_RESULT                    = 'results/sentiment_analysis_result_svm_polynomial.csv'
SENTIMENT_ANALYSIS_GRADIENT_BOOST_RESULT                    = 'results/sentiment_analysis_result_gradient_boost.csv'
SENTIMENT_ANALYSIS_LABEL_PROPAGATION_RESULT                 = 'results/sentiment_analysis_result_label_propagation.csv'
SENTIMENT_ANALYSIS_GAUSSIAN_NAIVE_BAYES_RESULT              = 'results/sentiment_analysis_result_gaussian_naive_bayes.csv'
SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT             = 'results/sentiment_analysis_result_multilayer_perceptron.csv'
SENTIMENT_ANALYSIS_LOGISTIC_RESULT_BY_STAT_FEAT             = 'results/sentiment_analysis_result_logistic_by_statistical_features.csv'
SENTIMENT_ANALYSIS_LIGHTGBM_RESULT_BY_STAT_FEAT             = 'results/sentiment_analysis_result_lightgbm_by_statistical_features.csv'
SENTIMENT_ANALYSIS_LOGISTIC_DECISION_TREE_RESULT            = 'results/sentiment_analysis_result_logistic_model_tree.csv'
SENTIMENT_ANALYSIS_MULTINOMIAL_NAIVE_BAYES_RESULT           = 'results/sentiment_analysis_result_naive_bayes.csv'
SENTIMENT_ANALYSIS_LOGISTIC_GAUSSIAN_NAIVE_BAYES_RESULT     = 'results/sentiment_analysis_result_logistic_gaussian_naive_bayes.csv'
SENTIMENT_ANALYSIS_HIST_GRADIENT_BOOSTING_CLASSIFIER_RESULT = 'results/sentiment_analysis_result_hist_gradient_boosting_classifier.csv'

# CONTEXTUAL FEATURE ENGINEERING PATHS
SENTIMENT_ANALYSIS_LOGISTIC_GAUSSIAN_NAIVE_BAYES_RESULT_WITH_CONTEXTUALS  = 'results/sentiment_analysis_result_logistic_gaussian_naive_bayes.csv'
SENTIMENT_ANALYSIS_LOGISTIC_DECISION_TREE_RESULT_WITH_CONTEXTUALS         = 'results/sentiment_analysis_result_logistic_model_tree_with_contextuals.csv'
SENTIMENT_ANALYSIS_MULTILAYER_PERCEPTRON_RESULT_WITH_CONTEXTUALS          = 'results/sentiment_analysis_result_multilayer_perceptron_with_contextuals.csv'
SENTIMENT_ANALYSIS_ADABOOST_RESULT_WITH_CONTEXTUALS                       = 'results/sentiment_analysis_result_adaboost_with_contextuals.csv'
SENTIMENT_ANALYSIS_SVM_RBF_RESULT_WITH_CONTEXTUALS                        = 'results/sentiment_analysis_result_svm_rbf_with_contextuals.csv'
SENTIMENT_ANALYSIS_GAUSSIAN_NAIVE_BAYES_RESULT_WITH_CONTEXTUALS           = 'results/sentiment_analysis_gaussian_naive_bayes_with_contextuals.csv'

# SEMANTIC FEATURE ENGINEERING PATHS
SENTIMENT_ANALYSIS_SVM_RBF_BY_SEMANTIC_FEAT_RESULT          = 'results/sentiment_analysis_result_svm_rbf_by_semantic_features.csv'
SENTIMENT_ANALYSIS_SVM_SIGMOID_BY_SEMANTIC_FEAT_RESULT      = 'results/sentiment_analysis_result_svm_rbf_by_semantic_features.csv'
SENTIMENT_ANALYSIS_LOGISTIC_REG_BY_SEMANTIC_FEAT_RESULT     = 'results/sentiment_analysis_result_logistic_reg_by_semantic_features.csv'
SENTIMENT_ANALYSIS_RANDOM_FOREST_BY_SEMANTIC_FEAT_RESULT    = 'results/sentiment_analysis_result_random_forest_by_semantic_features.csv'
SENTIMENT_ANALYSIS_LABEL_PROP_BY_SEMANTIC_FEAT_RESULT       = 'results/sentiment_analysis_result_label_prop_by_semantic_features.csv'

# CONFIGURATION VARIABLES
BATCH_SIZE                                                  = 250
MAX_FEATURES                                                = 1000
MODEL_NAME                                                  = "svm"
KERNEL_NAME                                                 = "sigmoid" # IF MODEL DOESN'T HAVE KERNEL, MAKE IT NONE
GLOVE_MODEL_PATH                                            = "models/glove.6B.100d.txt"
ELMO_MODEL_URL                                              = "https://tfhub.dev/google/elmo/3"


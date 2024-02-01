"""Predictive performance
This script will test the predictive performance of the data sets.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_file', type=str, default="none")
args = parser.parse_args()


# %% evaluate a model
def evaluate_model(x_train, x_test, y_train, y_test):
    """Evaluatation

    Args:
        x_train (pd.DataFrame): dataframe for train
        x_test (pd.DataFrame): dataframe for test
        y_train (np.int64): target variable for train
        y_test (np.int64): target variable for test
    Returns:
        tuple: dictionary with validation, train and test results
    """
    
    seed = np.random.seed(1234)

    # initiate models
    rf = RandomForestClassifier(random_state=seed)
    booster = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=seed)
    reg = LogisticRegression(random_state=seed)

    # set parameterisation
    param1 = {}
    param1['classifier__n_estimators'] = [100, 250, 500]
    param1['classifier__max_depth'] = [4, 7, 10]
    param1['classifier'] = [rf]

    param2 = {}
    param2['classifier__n_estimators'] = [100, 250, 500]
    param2['classifier__max_depth'] = [4, 7, 10]
    param2['classifier__learning_rate'] = [0.1, 0.01]
    param2['classifier'] = [booster]

    param3 = {}
    param3['classifier__C'] = np.logspace(-4, 4, 3)
    param3['classifier__max_iter'] = [1000000, 100000000]
    param3['classifier'] = [reg]

    # define metric functions
    scoring = {
        'gmean': make_scorer(geometric_mean_score),
        'acc': 'accuracy',
        'bal_acc': 'balanced_accuracy',
        'f1': 'f1',
        'f1_weighted': 'f1_weighted',
        'roc_auc_curve': make_scorer(roc_auc_score, max_fpr=0.001, needs_proba=True),
        }

    pipeline = Pipeline([('classifier', rf)])
    params = [param1, param2, param3]

    print("Start modeling with CV")
    # Train the grid search model
    gs = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='acc',
        return_train_score=True,
        n_jobs=-1).fit(x_train, y_train)

    score_cv = {
    'params':[], 'model':[],
    'test_accuracy': [], 'test_f1_weighted':[], 'test_gmean':[], 'test_roc_auc':[]
    }
    # Store results from grid search
    validation = pd.DataFrame(gs.cv_results_)
    validation['model'] = validation['param_classifier']
    validation['model'] = validation['model'].apply(lambda x: 'Random Forest' if 'RandomForest' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'XGBoost' if 'XGB' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Logistic Regression' if 'Logistic' in str(x) else x)

    print("Start modeling in out of sample")

    for i in range(len(validation)):
        # set each model for prediction on test
        clf_best = gs.best_estimator_.set_params(**gs.cv_results_['params'][i]).fit(x_train, y_train)
        clf = clf_best.predict(x_test)
        score_cv['params'].append(str(gs.cv_results_['params'][i]))
        score_cv['model'].append(validation.loc[i, 'model'])
        score_cv['test_accuracy'].append(accuracy_score(y_test, clf))
        score_cv['test_f1_weighted'].append(f1_score(y_test, clf, average='weighted'))
        score_cv['test_gmean'].append(geometric_mean_score(y_test, clf))
        score_cv['test_roc_auc'].append(roc_auc_score(y_test, clf))


    score_cv = pd.DataFrame(score_cv)

    return [validation, score_cv]


def save_results(file, results):
    """Create a folder if dooes't exist and save results

    Args:
        file (string): file name
        results (list of Dataframes): results for cross validation and out of sample
    """
    output_folder_val = ('modeling_results/validation')
    output_folder_test = ('modeling_results/test')
    if not os.path.exists(output_folder_val): os.makedirs(output_folder_val)
    if not os.path.exists(output_folder_test): os.makedirs(output_folder_test)

    results[0].to_csv(f'{output_folder_val}/{file}', index=False)
    results[1].to_csv(f'{output_folder_test}/{file}', index=False)



test_data = pd.read_csv('ds38_test.csv')
data = pd.read_csv(f'synth_data/{args.input_file}')

# prepare data to modeling
test_data = test_data.apply(LabelEncoder().fit_transform)
data = data.apply(LabelEncoder().fit_transform)

x_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]

x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

results = evaluate_model(x_train, x_test, y_train, y_test)

save_results(args.input_file, results)
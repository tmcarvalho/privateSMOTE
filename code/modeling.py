"""Predictive performance
This script will test the predictive performance of the data sets.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
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
    param3['classifier__max_iter'] = [1000000, 10000000]
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

    # Train the grid search model
    gs = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='f1_weighted',
        return_train_score=True,
        n_jobs=-1).fit(x_train, y_train)

    validation = {}

    # Store results from grid search
    validation = gs.cv_results_

    score = {
    'model':[],
    'test_f1_weighted':[], 'test_gmean':[], 'test_roc_auc':[]
    }
    
    # apply best cv result in all training data (without CV - out of sample)
    retrain = gs.best_estimator_.named_steps['classifier'].fit(x_train, y_train)
    # TODO in future: retrain with all models with best parameters in CV
    score['model'] = {gs.best_estimator_.named_steps["classifier"]}
    
    # Predict on test data with best params
    y_pred = retrain.predict(x_test)
    # Store predicted results
    score['test_f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    score['test_gmean'] = geometric_mean_score(y_test, y_pred)
    score['test_roc_auc'] = roc_auc_score(y_test, y_pred)
    
    return validation, score

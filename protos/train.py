# %load train.py

# Data prep libs
import pandas as pd
import numpy as np
import gc

# sklearn libs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score

# logging and loading libs
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from load_data import load_test_data, load_train_data

DIR = 'result_tmp/'
logger = getLogger(__name__)

if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df = load_train_data()
    x_train = df.drop(['target'], axis = 1)
    y_train = df['target']

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('Data prep ended {}'.format(x_train.shape))

    del df
    gc.collect()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    # All params of Logistic Regression
    #LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #      intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    #      penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
    #      verbose=0, warm_start=False)
    
    all_params= {
        
        'C': [10**i for i in range(-1, 2)],
        'fit_intercept': [True, False],
        'penalty': ['l2', 'l1'],
        'random_state': [0]
        
    }
    
    min_score = 100
    min_params = None
    
    #kfolds = StratifiedKFold(5)
    #clf = GridSearchCV(estimator, parameters, scoring=qwk, cv=kfolds.split(xtrain,ytrain))
    #clf.fit(xtrain, ytrain)

    
    clf = GridSearchCV(LogisticRegression(), all_params, scoring = 'roc_auc', cv=cv.split(x_train, y_train))
    clf.fit(x_train, y_train)
    best_params = clf.best_params_
    
    logger.info('CV results: {}'.format(clf.cv_results_))
    logger.info('Best score: {}'.format(clf.best_score_))
    logger.info('Best params: {}'.format(best_params))

    
    clf = LogisticRegression(**best_params)
    clf.fit(x_train, y_train)

    df = load_test_data()

    x_test = df[use_cols].sort_values('id')
    
    del df
    gc.collect()
    
    logger.info('Load test end')

    pred_test = clf.predict_proba(x_test)[:, 1]

    submission = pd.DataFrame(

        {'id': x_test['id'],
         'target': pred_test
        }
    )
    logger.info('Data frame successfully created')

    submission.to_csv(DIR + 'submission_lg.csv', index = False)

    logger.info('submission file written')
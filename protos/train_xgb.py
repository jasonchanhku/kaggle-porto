# Data prep libs
import pandas as pd
import numpy as np
import gc

# sklearn libs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
import xgboost as xgb

#viz
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# logging and loading libs
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from load_data import load_test_data, load_train_data

DIR = 'result_tmp/'
logger = getLogger(__name__)

def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    # in short, maximizing AUC also maximizes gini
    g = 2 * auc(fpr, tpr) - 1
    

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
        
    # All params of Logistic Regression
    #LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #      intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    #      penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
    #      verbose=0, warm_start=False)
    
    # Params of XGBClassifier()
    #xgb.sklearn.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic',               booster='gbtree', n_jobs=1,            nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,           colsample_bylevel=1, reg_alpha=0, reg_lambda=1,                     scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None,       **kwargs)
    
    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.3085,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.01,
        'objective': 'binary:logistic',
        'max_depth': 7,
        'num_parallel_tree': 1,
        'min_child_weight': 4.2922,
        'eval_metric': 'auc',
        'eta':0.1,
        'gamma': 0.5290,
        'subsample':0.9930,
        'max_delta_step':0,
        'booster':'gbtree',
        'nrounds': 1001
    }
    
    #
    #kfolds = StratifiedKFold(5)
    #clf = GridSearchCV(estimator, parameters, scoring=qwk, cv=kfolds.split(xtrain,ytrain))
    #clf.fit(xtrain, ytrain)

    # might use parametergrid
    
    x_train = xgb.DMatrix(x_train, label=y_train)
    
    xgb_cv_res = xgb.cv(xgb_params, x_train, num_boost_round=2001, nfold=5, seed = 0, stratified=True,
             early_stopping_rounds=25, verbose_eval=50, show_stdv=False)
    
    xgb_cv_res.plot(y=['train-auc-mean', 'test-auc-mean'],grid=True, logx=True)
    plt.xlabel('Round')
    plt.ylabel('AUC')
    plt.savefig('performance.png', dpi = 200, bbox_inches='tight')
    plt.show()
    
    best_nrounds = xgb_cv_res.shape[0] - 1
    xgb_best = xgb.train(xgb_params, x_train, best_nrounds)
    
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(xgb_best, max_num_features=50, height=0.8, ax=ax)
    plt.savefig('importance.png', bbox_inches='tight')
    plt.show()
    
    df = load_test_data()
    x_test = df[use_cols].sort_values('id')
    del df
    gc.collect()
    
    d_test = xgb.DMatrix(x_test)
    pred_test = xgb_best.predict(d_test)
    
    submission = pd.DataFrame(

        {'id': x_test['id'],
         'target': pred_test
        }
    )
    logger.info('Data frame successfully created')

    submission.to_csv(DIR + 'submission_xgb.csv', index = False)

    logger.info('submission file written')

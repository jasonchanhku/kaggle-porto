# Data prep libs
import pandas as pd
import numpy as np
import gc

# sklearn libs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
import lightgbm as lgb

#viz
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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



    # data_prep
    df = load_train_data()
    x_train = df.drop(['target'], axis = 1)
    y_train = df['target']

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('Data prep ended {}'.format(x_train.shape))

    del df
    gc.collect()

    df = load_test_data()

    x_test = df[use_cols].sort_values('id')

    del df
    gc.collect()

    NFOLDS = 5
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
    kf = kfold.split(x_train,y_train )

    # specify your configurations as a dict
    gridParams = {
        'learning_rate': [0.1, 0.05],
        'num_leaves': [15, 63],
        'boosting_type' : ['gbdt'],
        'objective' : ['binary'],
        'reg_alpha': [0.0, 0.1]

    }

    mdl = lgb.LGBMClassifier(
        boosting_type='gbdt', 
        num_leaves=31, 
        max_depth=-1, 
        learning_rate=0.1, 
        n_estimators=5000, 
        max_bin=255, 
        subsample_for_bin=50000, 
        objective='binary', 
        min_split_gain=0, 
        min_child_weight=5, 
        min_child_samples=10, 
        subsample=1, 
        subsample_freq=1, 
        colsample_bytree=1, 
        reg_alpha=0, 
        reg_lambda=0, 
        scale_pos_weight=16, 
        is_unbalance=False, 
        seed=218, 
        nthread=-1, 
        silent=True, 
        sigmoid=1.0, 
        drop_rate=0.1, 
        skip_drop=0.5, 
        max_drop=50, 
        uniform_drop=False, 
        xgboost_dart_mode=False    
    )

    scoring = {'AUC': 'roc_auc'}
    # Create the grid
    grid = GridSearchCV(mdl, gridParams, verbose=2, cv=kf, scoring=scoring, refit='AUC')
    # Run the grid
    grid.fit(x_train, y_train)    
    print('Best parameters found by grid search are:', grid.best_params_)
    print('Best score found by grid search is:', grid.best_score_)
    pred_test = grid.predict_proba(x_test)[:, 1]
    
    
    submission = pd.DataFrame(

        {'id': x_test['id'],
         'target': pred_test
        }
    )
    
    logger.info('Data frame successfully created')

    submission.to_csv(DIR + 'submission_lgbm_tuned.csv', index = False)

    logger.info('submission file written')
    

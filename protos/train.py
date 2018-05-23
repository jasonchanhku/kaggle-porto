# %load train.py

# Data prep libs
import pandas as pd
import numpy as np
import gc

# sklearn libs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
    list_auc_score = []
    list_logloss_score = []
    
    for train_idx, valid_idx in cv.split(x_train, y_train):
        
        trn_x = x_train.iloc[train_idx, :]
        val_x = x_train.iloc[valid_idx, :]
        
        trn_y = y_train.iloc[train_idx]
        val_y = y_train.iloc[valid_idx]
        
        clf = LogisticRegression(random_state=0)
        clf.fit(trn_x, trn_y)
        pred = clf.predict_proba(val_x)[:, 1]
        
        sc_logloss = log_loss(val_y, pred)
        sc_auc = roc_auc_score(val_y, pred)
        
        list_logloss_score.append(sc_logloss)
        list_auc_score.append(sc_auc)
        logger.debug('  logloss: {}, auc: {}'.format(sc_logloss, sc_auc))
        
    logger.info('Mean of CV: logloss: {}, auc: {}'.format(np.mean(list_logloss_score), np.mean(list_auc_score)))

    
    logger.info('train end')
    
    clf = LogisticRegression(random_state=0)
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
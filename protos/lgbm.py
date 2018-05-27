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
    learning_rate = 0.1
    num_leaves = 15
    min_data_in_leaf = 5000
    # max_bin = x
    feature_fraction = 0.6
    num_boost_round = 2000

    best_trees = []
    fold_scores = []
    evals_dict = {}
    cv_pred = 0


    params = {"objective": "binary",
              "boosting_type": "gbdt",
              "learning_rate": learning_rate,
              "num_leaves": int(num_leaves),
               "max_bin": 256,
              "min_data_in_leaf": min_data_in_leaf,
              "feature_fraction": feature_fraction,
              "verbosity": 0,
              "seed": 218,
              "drop_rate": 0.1,
              "is_unbalance": False,
              "max_drop": 50,
              "min_child_samples": 10,
              "min_child_weight": 150,
              "min_split_gain": 0,
              "subsample": 0.9,
              "metric": 'auc',
              "scale_pos_weight": 16
              }


    for i, (trn_idx, val_idx) in enumerate(kf):

        # getting indexes of cross validation
        x_trn = x_train.values[trn_idx, :]
        x_val = x_train.values[val_idx, :]
        y_trn = y_train.values[trn_idx]
        y_val = y_train.values[val_idx]

        # creating lgb datasets
        d_train = lgb.Dataset(x_trn, y_trn)
        d_val = lgb.Dataset(x_val, y_val)

        # model training
        bst = lgb.train(params, d_train, num_boost_round, valid_sets=[d_train, d_val], valid_names=['train', 'valid'], 
                        early_stopping_rounds=100, verbose_eval=100, evals_result=evals_dict)
        best_trees.append(bst.best_iteration)

        # predict on test set
        cv_pred += bst.predict(x_test, num_iteration=bst.best_iteration)


    # Latest evals dict
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(evals_dict['train']['auc'])
    ax.plot(evals_dict['valid']['auc'])
    plt.savefig('performance_lgb.png', bbox_inches='tight')
    plt.show()

    submission = pd.DataFrame(

        {'id': x_test['id'],
         'target': cv_pred/NFOLDS
        }
    )
    logger.info('Data frame successfully created')

    submission.to_csv(DIR + 'submission_lgb.csv', index = False)

    logger.info('submission file written for lgb')
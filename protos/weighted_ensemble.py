import pandas as pd
import numpy as np

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

    handler = FileHandler(DIR + 'weighted_ensemble.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')
    
    # all results
    lg = pd.read_csv(DIR + 'submission_lg.csv')
    xgb = pd.read_csv(DIR + 'submission_xgb.csv')
    lgb = pd.read_csv(DIR + 'submission_lgb.csv')
    
    id = lg['id']
    
    submission_weighted = pd.DataFrame({
        'id':id,
        'target':0.5*lgb['target'] + 0.25*xgb['target'] + 0.25*lg['target']

    })
    
    
    submission_weighted.to_csv(DIR + 'weighted_ensemble.csv', index = False)
    
    logger.info('Weighted ensemble complete and csv file generated')

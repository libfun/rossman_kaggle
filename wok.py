import numpy as np
import pandas as pd

from sklearn import cross_validation
import xgboost as xgb

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

train_a7 = pd.read_csv('/Users/libfun/1/train_a7.csv', index_col=None)

features = [u'Open', u'Promo', u'StateHoliday0',
       u'StateHolidaya', u'DayOfWeek1', u'DayOfWeek2', u'DayOfWeek3',
       u'DayOfWeek4', u'DayOfWeek5', u'DayOfWeek6', u'DayOfWeek7',
       u'CompetitionDistance', u'Promo2', 'year', 'MeanSales', 'month', 'day',
       u'StoreTypea', u'StoreTypeb', u'StoreTypec', u'StoreTyped',
       u'Assortmenta', u'Assortmentb', u'Assortmentc', u'CompetitionOpen']

ts_features = [u'Open', u'Promo', u'StateHoliday0',
       u'StateHolidaya', u'DayOfWeek1', u'DayOfWeek2', u'DayOfWeek3',
       u'DayOfWeek4', u'DayOfWeek5', u'DayOfWeek6', u'DayOfWeek7',
       u'CompetitionDistance', u'Promo2', 'year', 'MeanSales', 'month', 'day', u'CompetitionOpen']

allfs = features + [i+'mean' for i in ts_features] + [i+'std' for i in ts_features] + [i+'dta' for i in ts_features]

from hyperopt import hp, fmin, tpe

X_train, X_test = cross_validation.train_test_split(train_a7, test_size=0.05, random_state = 1)
dtrain = xgb.DMatrix(X_train[X_train['Open'] > 0][allfs], np.log(X_train[X_train['Open'] > 0]["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[X_test['Open'] > 0][allfs], np.log(X_test[X_test['Open'] > 0]["Sales"] + 1))

def calc(params):

    print params
    
    num_trees = 5000
    
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, 
                    feval=rmspe_xg, verbose_eval=False)

    #print("Validating")
    train_probs = gbm.predict(dvalid)
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[X_test['Open'] > 0]['Sales'].values)
    print 'Error:', error
    return {'loss': error, 'status': STATUS_OK}

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing

import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def score(params):
    print "Training with params : "
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round)
    predictions = model.predict(dvalid).reshape((X_test.shape[0], 9))
    score = log_loss(y_test, predictions)
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize():
    space = {
             'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 20, 1),
             'subsample' : hp.quniform('subsample', 0.2, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.2, 1, 0.05),
             'alpha' : hp.quniform('alpha', 0, 1, 0.05),
             'lambda' : hp.quniform('lambda', 0, 1000, 1),
             'objective': 'reg:linear',
             'silent' : 1
             }

    best = fmin(calc, space, algo=tpe.suggest, max_evals=2500)

    print best
    return best


#Trials object where the history of search will be stored
#trials = Trials()

best = optimize()

print 'Best result a7:', best

X_train, X_test = cross_validation.train_test_split(train_a18, test_size=0.05, random_state = 1)
dtrain = xgb.DMatrix(X_train[X_train['Open'] > 0][allfs], np.log(X_train[X_train['Open'] > 0]["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[X_test['Open'] > 0][allfs], np.log(X_test[X_test['Open'] > 0]["Sales"] + 1))

def calc(params):

    print params
    
    num_trees = 5000
    
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, 
                    feval=rmspe_xg, verbose_eval=False)

    #print("Validating")
    train_probs = gbm.predict(dvalid)
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[X_test['Open'] > 0]['Sales'].values)
    print 'Error:', error
    return {'loss': error, 'status': STATUS_OK}

def score(params):
    print "Training with params : "
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round)
    predictions = model.predict(dvalid).reshape((X_test.shape[0], 9))
    score = log_loss(y_test, predictions)
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize():
    space = {
             'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 20, 1),
             'subsample' : hp.quniform('subsample', 0.2, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.2, 1, 0.05),
             'alpha' : hp.quniform('alpha', 0, 1, 0.05),
             'lambda' : hp.quniform('lambda', 0, 1000, 1),
             'objective': 'reg:linear',
             'silent' : 1
             }

    best = fmin(calc, space, algo=tpe.suggest, max_evals=2500)

    print best
    return best


#Trials object where the history of search will be stored
#trials = Trials()

best = optimize()

print 'Best result a18:', best

X_train, X_test = cross_validation.train_test_split(train_a30, test_size=0.05, random_state = 1)
dtrain = xgb.DMatrix(X_train[X_train['Open'] > 0][allfs], np.log(X_train[X_train['Open'] > 0]["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[X_test['Open'] > 0][allfs], np.log(X_test[X_test['Open'] > 0]["Sales"] + 1))

def calc(params):

    print params
    
    num_trees = 5000
    
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, 
                    feval=rmspe_xg, verbose_eval=False)

    #print("Validating")
    train_probs = gbm.predict(dvalid)
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[X_test['Open'] > 0]['Sales'].values)
    print 'Error:', error
    return {'loss': error, 'status': STATUS_OK}

def score(params):
    print "Training with params : "
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round)
    predictions = model.predict(dvalid).reshape((X_test.shape[0], 9))
    score = log_loss(y_test, predictions)
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize():
    space = {
             'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 20, 1),
             'subsample' : hp.quniform('subsample', 0.2, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.2, 1, 0.05),
             'alpha' : hp.quniform('alpha', 0, 1, 0.05),
             'lambda' : hp.quniform('lambda', 0, 1000, 1),
             'objective': 'reg:linear',
             'silent' : 1
             }

    best = fmin(calc, space, algo=tpe.suggest, max_evals=2500)

    print best
    return best


#Trials object where the history of search will be stored
#trials = Trials()

best = optimize()

print 'Best result a30:', best

import sys
import os
import random
import math

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold

TRAIN_FRAC = 0.9

inst = sys.argv[1]
fn = sys.argv[2]
seed = max(0,int(sys.argv[3]))
config = sys.argv[4:]

if not(inst.startswith("i") or inst.startswith("test")):
    raise ValueError("instance has to be i%d or test")

np.random.seed(12345)

# read in data
print("Load from %s " % (fn))

data = np.loadtxt(fn,delimiter=",",dtype=np.int)

np.random.shuffle(data)
y = np.array(data[:, -1], dtype=np.int)
X = np.array(data[:, :-1], dtype=np.float)

del data

n_train = math.ceil(X.shape[0] * TRAIN_FRAC)

X_train = X[:n_train, :]
y_train = y[:n_train]

X_test = X[n_train:, :]
y_test = y[n_train:]

# if instance starts with "i", use i-th cv-split on trainings data
if inst.startswith("i"):
    fold_indx = int(inst[1:]) - 1
    print("Use %dth-cv split" %(fold_indx))
    kfold = KFold(n_splits=10, shuffle=False, random_state=12345)
    kf = list(kfold.split(X=X_train))
    train_index, test_index = kf[fold_indx]
    X_train_fold = X_train[train_index, :]
    y_train_fold = y_train[train_index]
    X_test_fold = X_train[test_index, :]
    y_test_fold = y_train[test_index]
    X_train, y_train = X_train_fold, y_train_fold
    X_test, y_test = X_test_fold, y_test_fold
    
# set params of xgboost
param = {'random_state' : seed, "cache_size": 1000}

iter_conf = iter(config)
int_params = ["degree"]
str_params = ["kernel"]
bool_params = ["shrinking"]
while True:
    try:
        name = next(iter_conf).strip("-")
        value = next(iter_conf)
        if name in int_params:
            param[name] = int(value)
        elif name in str_params:
            param[name] = str(value)
        elif name in bool_params:
            if value == "True":
                param[name] = True
            else:
                param[name] = False
        else:
            param[name] = float(value)
    except StopIteration:
        break


# Make sure inactive params are not present
if param['kernel'] == "rbf":
    try:
        del param["degree"]
    except:
        pass
    try:
        del param["coef0"]
    except:
        pass

if param["kernel"] == "sigmoid":
    try:
        del param["degree"]
    except:
        pass

print(param)

print("Train on %d points" %(y_train.shape[0]))
svc = SVC(**param)
svc.fit(X_train, y_train)
preds = svc.predict(X_test)

print(preds)

acc = float(np.sum(preds == y_test)) / y_test.shape[0]

print("Error: %.6f" %(1-acc))

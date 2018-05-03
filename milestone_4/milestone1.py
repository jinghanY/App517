import numpy as np
import os
import time

from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, DotProduct, Matern)
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel as C

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import normalize
from scipy.stats import zscore

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree


def readData(fileName):
    f = open(fileName, 'r')
    lines = f.readlines()
    header = lines[0]
    header_ele = header.split(';')
    header_ele = [x.strip('"') for x in header_ele]
    header_ele = header_ele[:-1]
    data_feature = []
    ct = 0
    for line in lines:
        ct = ct + 1
        if ct == 1:
            continue
        element = line.split(';')
        element = element[:-1]
        features = [float(i) for i in element]
        data_feature.append(features)

    data_feature = np.array(data_feature)
    return data_feature, header_ele


def loadLabels(inFileName):
    inFile = open(inFileName, 'r')
    lines = inFile.readlines()
    label = []
    class_names = []
    for line in lines:
        a = int(float(line))
        if a == 1:
            class_names.append('yes')
        else:
            class_names.append('no')
        label.append(a)
    label = np.array(label)
    return label, class_names



# Set elapsed time, random seed, number of folds, etc.
elapsed = time.time()
np.random.seed(42)
n_splits = 10


# Read data
#print("Start loading data @ %.5f\n" % (time.time()-elapsed))
cwd = os.getcwd()
file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"
#print("End loading data @ %.5f\n" % (time.time()-elapsed))

#print("Start shuffling and sampling @ %.5f\n" % (time.time()-elapsed))
features, header_ele = readData(file_name_feature)
features = shuffle(features, random_state=41)[:5000]

features[:, :9] = zscore(features[:, :9])

label, label_names = loadLabels(file_name_label)
label = shuffle(label, random_state=41)[:5000]
#print("End shuffling and sampling @ %.5f\n" % (time.time()-elapsed))

#print("Start splitting dataset with 10-folds @ %.5f\n" % (time.time()-elapsed))
Kfold = StratifiedKFold(n_splits=n_splits)
#print("End splitting dataset with 10-folds @ %.5f\n" % (time.time()-elapsed))

accuracy_training_log = np.zeros(n_splits)
accuracy_testing_log = np.zeros(n_splits)
nlpd_t = np.zeros(n_splits)
nlpd_v = np.zeros(n_splits)

for i, (train_index, test_index) in enumerate(Kfold.split(features, label)):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    model = LogisticRegression(penalty = 'l1')
    model.fit(X_train, y_train)
    accuracy_training_log[i] = accuracy_score(y_train, model.predict(X_train))
    accuracy_testing_log[i] = accuracy_score(y_test, model.predict(X_test))
    neg_lpd_t = -np.mean(np.log(model.predict_proba(X_train)[np.arange(len(X_train)), y_train]))
    neg_lpd_v = -np.mean(np.log(model.predict_proba(X_test)[np.arange(len(X_test)), y_test]))
    nlpd_t[i] = neg_lpd_t
    nlpd_v[i] = neg_lpd_v

print("The average training accuracy of Logistic regression is " + str(np.mean(accuracy_training_log)) + " and the average testing accuracy is " + str(np.mean(accuracy_testing_log)))
print("Average negative log predictive density of training set: %.5f"
      % np.mean(nlpd_t))
print("Average negative log predictive density of validation set: %.5f"
      % np.mean(nlpd_v))
print("Total elapsed time: %.5f" % (time.time()-elapsed))
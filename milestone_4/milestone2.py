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

cwd = os.getcwd()
file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"

features, header_ele = readData(file_name_feature)
features = shuffle(features, random_state=41)[:5000]
features[:, :9] = zscore(features[:, :9])

label, label_names = loadLabels(file_name_label)
label = shuffle(label, random_state=41)[:5000]
Kfold = StratifiedKFold(n_splits=n_splits)

accuracy_rbf_training = np.zeros(n_splits)
accuracy_rbf_testing = np.zeros(n_splits)
accuracy_matern_traing = np.zeros(n_splits)
accuracy_matern_testing = np.zeros(n_splits)
nlpd_rbf_t = np.zeros(n_splits)
nlpd_matern_t = np.zeros(n_splits)
nlpd_rbf_v = np.zeros(n_splits)
nlpd_matern_v = np.zeros(n_splits)
best_kernel = None
best_nlpd = np.inf

for i, (train_index, test_index) in enumerate(Kfold.split(features, label)):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    gp_rbf_fix = GaussianProcessClassifier(kernel=76.5**2 * RBF(length_scale=179),
                                           optimizer=None)
    gp_matern_fix = GaussianProcessClassifier(kernel=3.7**2 * Matern(length_scale=9.4, nu=1.5),
                                              optimizer=None)
    gp_rbf_fix.fit(X_train, y_train)
    gp_matern_fix.fit(X_train,y_train)
    accuracy_rbf_training[i] = accuracy_score(y_train, gp_rbf_fix.predict(X_train))
    accuracy_matern_traing[i] = accuracy_score(y_train, gp_matern_fix.predict(X_train))
    accuracy_rbf_testing[i] = accuracy_score(y_test, gp_rbf_fix.predict(X_test))
    accuracy_matern_testing[i] = accuracy_score(y_test, gp_matern_fix.predict(X_test))
print("Average training accuracy with rbf kernel: %.5f" % np.mean(accuracy_rbf_training))
print("Average testing accuracy with rbf kernel: %.5f" % np.mean(accuracy_rbf_testing))
print("Average training accuracy with matern kernel: %.5f" % np.mean(accuracy_matern_traing))
print("Average testing accuracy with matern kernel: %.5f" % np.mean(accuracy_matern_testing))
print("Total elapsed time: %.5f" % (time.time()-elapsed))

import numpy as np
import os
import time

from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from scipy.stats import zscore
from sklearn import svm
from sklearn import metrics



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
print("Start loading data @ %.5f\n" % (time.time()-elapsed))
cwd = os.getcwd()
file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"
print("End loading data @ %.5f\n" % (time.time()-elapsed))

print("Start shuffling and sampling @ %.5f\n" % (time.time()-elapsed))
features, header_ele = readData(file_name_feature)
features = shuffle(features, random_state=42)[:5000]
features[:, :9] = zscore(features[:, :9])

label, label_names = loadLabels(file_name_label)
label = shuffle(label, random_state=42)[:5000]
print("End shuffling and sampling @ %.5f\n" % (time.time()-elapsed))

# Use GridSearchCV to find the best parameters for C and gamma of rbf kernel.

# print("Start searching for best parameters @ %.5f\n" % (time.time()-elapsed))
# C_range = np.logspace(-0, 2, 1)
# gamma_range = np.logspace(-1, 1, 1)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
# grid.fit(features, label)
# print("The best parameters are %s with a score of %0.2f\n"
#       % (grid.best_params_, grid.best_score_))
# # The best parameters are {'gamma': 0.10000000000000001, 'C': 1.0} with a score of 0.90
# print("End searching for best parameters @ %.5f\n" % (time.time()-elapsed))

print("Start splitting dataset with 10-folds @ %.5f\n" % (time.time()-elapsed))
Kfold = StratifiedKFold(n_splits=n_splits)
print("End splitting dataset with 10-folds @ %.5f\n" % (time.time()-elapsed))

accuracy_test = np.zeros(n_splits)
accuracy_train = np.zeros(n_splits)
f1 = np.zeros(n_splits)
for i, (train_index, test_index) in enumerate(Kfold.split(features, label)):
    print("Start training model %d @ %.5f\n" % (i, time.time()-elapsed))
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # SVC with rbf kernel - C=1, gamma = 0.1
    svm_model = svm.SVC(C=1, gamma=0.1)

    # SVC with linear kernel
    # svm_model = svm.LinearSVC()
    svm_model.fit(X_train, y_train)
    print("End training model %d @ %.5f\n" % (i, time.time()-elapsed))

    accuracy_test[i] = accuracy_score(y_test, svm_model.predict(X_test))
    accuracy_train[i] = accuracy_score(y_train, svm_model.predict(X_train))
    f1[i] = metrics.f1_score(y_train, svm_model.predict(X_train))
    tn, fp, fn, tp = metrics.confusion_matrix(y_train, svm_model.predict(X_train)).ravel()
    print("TNR, FNR = %.5f, %.5f" % (tn/(tn+fp), fn/(tp+fn)))
    print("F1-score: %.5f" % f1[i])
    print("Accuracy for X_train: %.5f" % accuracy_train[i])
    print("Accuracy for X_test: %.5f\n" % accuracy_test[i])

print("Average accuracy for X_test %.5f\n" % np.mean(accuracy_test))

print("Total elapsed time: %.5f" % (time.time()-elapsed))

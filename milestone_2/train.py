import numpy as np
import os
import time

from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, DotProduct, Matern)
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel as C

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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

# Read data
print("Start loading data @ %.5f\n" % (time.time()-elapsed))
cwd = os.getcwd()
file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"
print("End loading data @ %.5f\n" % (time.time()-elapsed))

print("Start shuffling and sampling @ %.5f\n" % (time.time()-elapsed))
features, header_ele = readData(file_name_feature)
features = shuffle(features, random_state=42)[:500]
features[:, :9] = zscore(features[:, :9])

label, label_names = loadLabels(file_name_label)
label = shuffle(label, random_state=42)[:500]
print("End shuffling and sampling @ %.5f\n" % (time.time()-elapsed))

print("Start splitting dataset with 10-folds @ %.5f\n" % (time.time()-elapsed))
Kfold = StratifiedKFold(n_splits=n_splits)
print("End splitting dataset with 10-folds @ %.5f\n" % (time.time()-elapsed))

accuracy_rbf = np.zeros(n_splits)
accuracy_matern = np.zeros(n_splits)
nlpd_rbf_t = np.zeros(n_splits)
nlpd_matern_t = np.zeros(n_splits)
nlpd_rbf_v = np.zeros(n_splits)
nlpd_matern_v = np.zeros(n_splits)
best_kernel = None
best_nlpd = np.inf

for i, (train_index, test_index) in enumerate(Kfold.split(features, label)):
    print("Start training model %d @ %.5f\n" % (i, time.time()-elapsed))
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # Find best parameter for each kernel
    # kernel = C(0.1, (1e-5, np.inf)) * DotProduct(sigma_0=0.1) ** 2
    # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
    # kernel = 1.0 * RBF(length_scale=1.0)
    # gp_opt = GaussianProcessClassifier(kernel=kernel)
    # gp_opt.fit(X_train, y_train)
    # neg_lpd_opt = -np.mean(np.log(gp_opt.predict_proba(X_test)[np.arange(len(X_test)), y_test]))
    # print("Optimized kernel of model %d is %s @ %.5f\n" % (i, gp_opt.kernel_, time.time()-elapsed))
    # if neg_lpd_opt < best_nlpd:
    #     best_kernel = gp_opt.kernel_
    #     best_nlpd = neg_lpd_opt

    # Specify Gaussian Processes with fixed and optimized hyperparameters
    gp_rbf_fix = GaussianProcessClassifier(kernel=3.49 ** 2 * RBF(length_scale=5.28),
                                           optimizer=None)
    gp_matern_fix = GaussianProcessClassifier(kernel=3.84 ** 2 * Matern(length_scale=8.06, nu=1.5),
                                              optimizer=None)
    gp_rbf_fix.fit(X_train, y_train)
    gp_matern_fix.fit(X_train,y_train)
    print("End training model %d @ %.5f\n" % (i, time.time()-elapsed))

    # negative log predictive density
    neg_lpd_rbf_t = -np.mean(np.log(gp_rbf_fix.predict_proba(X_train)[np.arange(len(X_train)), y_train]))
    print("Negative log predictive density of training set with rbf kernel %.3f" % neg_lpd_rbf_t)
    neg_lpd_matern_t = -np.mean(np.log(gp_matern_fix.predict_proba(X_train)[np.arange(len(X_train)), y_train]))
    print("Negative log predictive density of training set with matern kernel %.3f" % neg_lpd_matern_t)
    neg_lpd_rbf_v = -np.mean(np.log(gp_rbf_fix.predict_proba(X_test)[np.arange(len(X_test)), y_test]))
    print("Negative log predictive density of validation set with rbf kernel %.3f" % neg_lpd_rbf_v)
    neg_lpd_matern_v = -np.mean(np.log(gp_matern_fix.predict_proba(X_test)[np.arange(len(X_test)), y_test]))
    print("Negative log predictive density of validation set with matern kernel %.3f" % neg_lpd_matern_v)
    nlpd_rbf_t[i] = neg_lpd_rbf_t
    nlpd_matern_t[i] = neg_lpd_matern_t
    nlpd_rbf_v[i] = neg_lpd_rbf_v
    nlpd_matern_v[i] = neg_lpd_matern_v

    accuracy_rbf[i] = accuracy_score(y_train, gp_rbf_fix.predict(X_train))
    print("Accuracy for X_train with rbf kernel: %.5f" % accuracy_rbf[i])
    print("Accuracy for X_test with rbf kernel: %.5f"
          % accuracy_score(y_test, gp_rbf_fix.predict(X_test)))

    accuracy_matern[i] = accuracy_score(y_train, gp_matern_fix.predict(X_train))
    print("Accuracy for X_train with matern kernel: %.5f" % accuracy_matern[i])
    print("Accuracy for X_test with matern kernel: %.5f\n"
          % accuracy_score(y_test, gp_matern_fix.predict(X_test)))

print("Average accuracy with rbf kernel: %.5f" % np.mean(accuracy_rbf))
print("Average accuracy with matern kernel: %.5f" % np.mean(accuracy_matern))
print("Average negative log predictive density of training set with rbf kernel: %.5f"
      % np.mean(nlpd_rbf_t))
print("Average negative log predictive density of training set with matern kernel: %.5f"
      % np.mean(nlpd_matern_t))
print("Average negative log predictive density of validation set with rbf kernel: %.5f"
      % np.mean(nlpd_rbf_v))
print("Average negative log predictive density of validation set with matern kernel: %.5f"
      % np.mean(nlpd_matern_v))

# print("Best kernel is ", best_kernel)
# print("Best negative log predictive density is ", best_nlpd)



# # Plot posteriors
# plt.figure(0)
# plt.scatter(X[:train_size, 0], y_train, c='k', label="Train data",
#             edgecolors=(0, 0, 0))
# plt.scatter(X[train_size:, 0], y[train_size:], c='g', label="Test data",
#             edgecolors=(0, 0, 0))
# X_ = np.linspace(0, 5, 100)
# plt.plot(X_, gp_fix.predict_proba(X_[:, np.newaxis])[:, 1], 'r',
#          label="Initial kernel: %s" % gp_fix.kernel_)
# plt.plot(X_, gp_opt.predict_proba(X_[:, np.newaxis])[:, 1], 'b',
#          label="Optimized kernel: %s" % gp_opt.kernel_)
# plt.xlabel("Feature")
# plt.ylabel("Class 1 probability")
# plt.xlim(0, 5)
# plt.ylim(-0.25, 1.5)
# plt.legend(loc="best")
#
# # Plot LML landscape
# plt.figure(1)
# theta0 = np.logspace(0, 8, 30)
# theta1 = np.logspace(-1, 1, 29)
# Theta0, Theta1 = np.meshgrid(theta0, theta1)
# LML = [[gp_opt.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
#         for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
# LML = np.array(LML).T
# plt.plot(np.exp(gp_fix.kernel_.theta)[0], np.exp(gp_fix.kernel_.theta)[1],
#          'ko', zorder=10)
# plt.plot(np.exp(gp_opt.kernel_.theta)[0], np.exp(gp_opt.kernel_.theta)[1],
#          'ko', zorder=10)
# plt.pcolor(Theta0, Theta1, LML)
# plt.xscale("log")
# plt.yscale("log")
# plt.colorbar()
# plt.xlabel("Magnitude")
# plt.ylabel("Length-scale")
# plt.title("Log-marginal-likelihood")
#
# plt.show()


print("Total elapsed time: %.5f" % (time.time()-elapsed))

import numpy as np
import os
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from sklearn.metrics.classification import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, DotProduct, Matern)

from sklearn import svm

from scipy.stats import zscore


def readData(fileName):
	f = open(fileName, "r")
	lines = f.readlines()
	header = lines[0]
	header_ele = header.split(";")
	header_ele = [x.strip('"') for x in header_ele]
	header_ele = header_ele[:-1]
	data_feature = []
	ct = 0
	for line in lines:
		ct = ct+1
		if ct == 1:
			continue
		element = line.split(";")
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
			class_names.append("yes")
		else:
			class_names.append("no")
		label.append(a)
	label = np.array(label)
	return label, class_names

def dimensionReduction(dataTable):
	# do SVD to the raw data
	u, s, vh = np.linalg.svd(dataTable, full_matrices=False)
	# remain the first and second principal component
	mutiplier = np.transpose(u[:,:2])
	# reconstruct the data matrix
	reconsData = mutiplier @ dataTable
	return reconsData

def linearClassify(X_train, y_train, X_test, y_test, iteration):
	model = LogisticRegression(penalty = 'l1')

	print("************ logistic regression with lasso ***********\n")
	start_train = time.time()
	model.fit(X_train, y_train)
	end_train = time.time()
	training_time = end_train - start_train
	print("Training linear model %d took %.5f\n" % (iteration, training_time))
	# predict class labels for the train set
	predicted_train = model.predict(X_train)
	print("training accuracy")
	print(metrics.accuracy_score(y_train,predicted_train))
	print("\n")

	# predict class labels for the test set
	start_test = time.time()
	predicted = model.predict(X_test)
	end_test = time.time()
	testing_time = end_test - start_test
	print("Testing linear model %d took %.5f\n" % (iteration, testing_time))
	print("testing accuracy")
	print(metrics.accuracy_score(y_test,predicted))
	print("\n")

	return training_time, testing_time

def gaussianProcess(X_train, y_train, X_test, y_test, iteration):
	print("************ Gaussian Process Classification **************\n")
	gp_rbf_fix = GaussianProcessClassifier(kernel=76.5**2 * RBF(length_scale=179), optimizer=None)
	start_train_gp = time.time()
	gp_rbf_fix.fit(X_train, y_train)
	end_train_gp = time.time()
	training_time_gp = end_train_gp - start_train_gp
	print("Training GP model_selection %d took %.5f\n" % (iteration, training_time_gp))

	predict_train_gp = gp_rbf_fix.predict(X_train)
	print("training accuracy")
	print(accuracy_score(y_train, predict_train_gp))
	print("\n")

	start_test_gp = time.time()
	predict_test_gp = gp_rbf_fix.predict(X_test)
	end_test_gp = time.time()
	testing_time_gp = end_test_gp - start_test_gp
	print("Testing GP model_selection %d took %.5f\n" % (iteration, training_time_gp))
	print("testing accuracy")
	print(accuracy_score(y_test, predict_test_gp))
	print("\n")

	return training_time_gp, testing_time_gp

def svmClassify(X_train, y_train, X_test, y_test, iteration):
	print("******************* SVM classification *********************\n")
	svm_model = svm.SVC(C=1, gamma=0.1)
	start_train_svm = time.time()
	svm_model.fit(X_train, y_train)
	end_train_svm = time.time()
	training_time_svm = end_train_svm - start_train_svm
	print("Training SVM model_selection %d took %.5f\n" % (iteration, training_time_svm))

	predict_train_svm = svm_model.predict(X_train)
	print("training accuracy")
	print(accuracy_score(y_train, predict_train_svm))
	print("\n")

	start_test_svm = time.time()
	predict_test_svm = svm_model.predict(X_test)
	end_test_svm = time.time()
	testing_time_svm = end_test_svm - start_test_svm
	print("Testing SVM model_selection %d took %.5f\n" % (iteration, testing_time_svm))
	print("testing accuracy")
	print(accuracy_score(y_test, predict_test_svm))
	print("\n")

	return training_time_svm, testing_time_svm

def signTest(train_time1, test_time1, train_time2, test_time2, model1, model2):
	print("*************** Comparing between ", model1, "and ", model2, " *****************\n")
	win1_train = 0
	win2_train = 0
	win1_test = 0
	win2_test = 0
	for n in range(0,10):
		if train_time1[n] < train_time2[n]:
			win1_train = win1_train + 1
		else:
			win2_train = win2_train + 1

		if test_time1[n] < test_time2[n]:
			win1_test = win1_test + 1
		else:
			win2_test = win2_test + 1

	if win1_train > 8:
		print(model1, " is better than ", model2, " w.r.t efficiency of training\n")
	else:
		if win2_train > 8:
			print(model2, " is better than ", model1, " w.r.t efficiency of training\n")
		else:
			print(model1, " and ", model2, " is similar w.r.t efficiency of training\n")

	if win1_test > 8:
		print(model1, " is better than ", model2, " w.r.t efficiency of testing\n")
	else:
		if win2_test > 8:
			print(model2, " is better than ", model1, " w.r.t efficiency of testing\n")
		else:
			print(model1, " and ", model2, " is similar w.r.t efficiency of testing\n")

np.random.seed(42)
n_splits = 10
Kfold = StratifiedKFold(n_splits=n_splits)

# load the raw data and the labels
cwd = os.getcwd()
file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"
dataset,header_ele = readData(file_name_feature)
label, class_names = loadLabels(file_name_label)

# shuffle and sample the dataset
dataset = shuffle(dataset, random_state=41)[:5000]
dataset[:, :9] = zscore(dataset[:, :9])
label = shuffle(label, random_state=41)[:5000]

train_time_l = np.zeros(n_splits)
test_time_l = np.zeros(n_splits)
train_time_gp = np.zeros(n_splits)
test_time_gp = np.zeros(n_splits)
train_time_svm = np.zeros(n_splits)
test_time_svm = np.zeros(n_splits)
train_time_svd = np.zeros(n_splits)
test_time_svd = np.zeros(n_splits)

for i, (train_index, test_index) in enumerate(Kfold.split(dataset, label)):
	X_train, X_test = dataset[train_index], dataset[test_index]
	y_train, y_test = label[train_index], label[test_index]
	train_time_l[i], test_time_l[i] = linearClassify(X_train, y_train, X_test, y_test, i)
	train_time_gp[i], test_time_gp[i] = gaussianProcess(X_train, y_train, X_test, y_test, i)
	train_time_svm[i], test_time_svm[i] = svmClassify(X_train, y_train, X_test, y_test, i)

print("************* final results for the efficiency of linear classification using raw data ****************\n")
print("Average training time of linear calssification: %.5f" % np.mean(train_time_l))
print("Average testing time of linear calssification: %.5f\n" % np.mean(test_time_l))

print("************* final results for the efficiency of GP classification using raw data ****************\n")
print("Average training time of GP calssification: %.5f" % np.mean(train_time_gp))
print("Average testing time of GP calssification: %.5f\n" % np.mean(test_time_gp))

print("************* final results for the efficiency of SVM classification using raw data ****************\n")
print("Average training time of SVM calssification: %.5f" % np.mean(train_time_svm))
print("Average testing time of SVM calssification: %.5f\n" % np.mean(test_time_svm))

# using SVD to do dimension reduction to raw data
dataset = np.transpose(dataset)
reconsData = dimensionReduction(dataset)
reconsData = np.transpose(reconsData)

for j, (train_index, test_index) in enumerate(Kfold.split(reconsData, label)):
	X_train, X_test = reconsData[train_index], reconsData[test_index]
	y_train, y_test = label[train_index], label[test_index]
	train_time_svd[j], test_time_svd[j] = gaussianProcess(X_train, y_train, X_test, y_test, j)

print("************* final results for the efficiency of GP classification using SVD data ****************\n")
print("Average training time of SVD GP calssification: %.5f" % np.mean(train_time_svd))
print("Average testing time of SVD GP calssification: %.5f\n" % np.mean(test_time_svd))

# doing sign test to train and test times for the 4 methods
signTest(train_time_l, test_time_l, train_time_gp, test_time_gp, "linear", "GP")
signTest(train_time_l, test_time_l, train_time_svm, test_time_svm, "linear", "SVM")
signTest(train_time_l, test_time_l, train_time_svd, test_time_svd, "linear", "SVD_GP")
signTest(train_time_gp, test_time_gp, train_time_svm, test_time_svm, "GP", "SVM")
signTest(train_time_gp, test_time_gp, train_time_svd, test_time_svd, "GP", "SVD_GP")
signTest(train_time_svm, test_time_svm, train_time_svd, test_time_svd, "SVM", "SVD_GP")












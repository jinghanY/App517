import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, DotProduct, Matern)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
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

def showFigure(reconsData, label):
	row, column = reconsData.shape
	i = 0
	for i in range(0,column):
		if label[i] == 0:
			plt.plot(reconsData[0,i],reconsData[1,i],'bx')
		else:
			plt.plot(reconsData[0,i],reconsData[1,i],'r+')
	plt.xlabel('principal component 1')
	plt.ylabel('principal component 2')
	plt.title('visualized dataset after SVD')
	plt.show()

start = time.time()
np.random.seed(42)
n_splits = 10
Kfold = StratifiedKFold(n_splits=n_splits)

cwd = os.getcwd()
file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"
dataset,header_ele = readData(file_name_feature)
label, class_names = loadLabels(file_name_label)

# shuffle and sample the data
dataset = shuffle(dataset, random_state=41)[:5000]
dataset[:, :9] = zscore(dataset[:, :9])
label = shuffle(label, random_state=41)[:5000]

raw_train_acc = np.zeros(n_splits)
raw_test_acc = np.zeros(n_splits)
nlpd_rbf_train_raw = np.zeros(n_splits)
nlpd_rbf_test_raw = np.zeros(n_splits)
svd_train_acc = np.zeros(n_splits)
svd_test_acc = np.zeros(n_splits)
nlpd_rbf_train_svd = np.zeros(n_splits)
nlpd_rbf_test_svd = np.zeros(n_splits)

# using the raw data to train and test the linear classification model
print("Using the raw dataset \n")
for i, (train_index, test_index) in enumerate(Kfold.split(dataset, label)):
	print("******************* Gaussian Process classification using raw data *****************\n")
	print("Start training model %d @ %.5f\n" % (i, time.time()-start))
	X_train, X_test = dataset[train_index], dataset[test_index]
	y_train, y_test = label[train_index], label[test_index]
	#train_accuracy, test_accuracy = linearClassify(X_train, y_train, X_test, y_test)
	gp_rbf_fix = GaussianProcessClassifier(kernel=76.5**2 * RBF(length_scale=179),optimizer=None)
	gp_rbf_fix.fit(X_train, y_train)

	neg_lpd_rbf_t = -np.mean(np.log(gp_rbf_fix.predict_proba(X_train)[np.arange(len(X_train)), y_train]))
	nlpd_rbf_train_raw[i] = neg_lpd_rbf_t
	print("Negative log predictive density of training set with rbf kernel of raw data %.3f" % neg_lpd_rbf_t)

	neg_lpd_rbf_v = -np.mean(np.log(gp_rbf_fix.predict_proba(X_test)[np.arange(len(X_test)), y_test]))
	nlpd_rbf_test_raw[i] = neg_lpd_rbf_v
	print("Negative log predictive density of validation set with rbf kernel of raw data %.3f" % neg_lpd_rbf_v)
	print("\n")

	print("End training model %d @ %.5f\n" % (i, time.time()-start))

	raw_train_acc[i] = accuracy_score(y_train, gp_rbf_fix.predict(X_train))
	print("Accuracy for X_train with rbf kernel of raw data: %.5f" % raw_train_acc[i])
	raw_test_acc[i] = accuracy_score(y_test, gp_rbf_fix.predict(X_test))
	print("Accuracy for X_test with rbf kernel of raw data: %.5f" % raw_test_acc[i])
	print("\n")

print("**************** final results using raw data ********************\n")
print("Average train accuracy of raw data: %.5f" % np.mean(raw_train_acc))
print("Average test accuracy of raw data: %.5f" % np.mean(raw_test_acc))
print("Average negative log predictive density of training set with rbf kernel of raw data: %.5f" % np.mean(nlpd_rbf_train_raw))
print("Average negative log predictive density of validation set with rbf kernel of raw data: %.5f" % np.mean(nlpd_rbf_test_raw))
print("\n\n")

dataset = np.transpose(dataset)
reconsData = dimensionReduction(dataset)
showFigure(reconsData, label)
reconsData = np.transpose(reconsData)

# using the data after SVD(which remain the first and second principal component) to train and test the linear classification model
print("Using the dataset after SVD \n")
for j, (train_index, test_index) in enumerate(Kfold.split(reconsData, label)):
	print("******************* Gaussian Process classification using svd data *****************\n")
	print("Start training model %d @ %.5f\n" % (j, time.time()-start))
	X_train, X_test = reconsData[train_index], reconsData[test_index]
	y_train, y_test = label[train_index], label[test_index]
	#train_accuracy, test_accuracy = linearClassify(X_train, y_train, X_test, y_test)
	gp_rbf_fix = GaussianProcessClassifier(kernel=76.5**2 * RBF(length_scale=179),optimizer=None)
	gp_rbf_fix.fit(X_train, y_train)

	neg_lpd_rbf_t = -np.mean(np.log(gp_rbf_fix.predict_proba(X_train)[np.arange(len(X_train)), y_train]))
	nlpd_rbf_train_svd[j] = neg_lpd_rbf_t
	print("Negative log predictive density of training set with rbf kernel of svd data %.3f" % neg_lpd_rbf_t)

	neg_lpd_rbf_v = -np.mean(np.log(gp_rbf_fix.predict_proba(X_test)[np.arange(len(X_test)), y_test]))
	nlpd_rbf_test_svd[j] = neg_lpd_rbf_v
	print("Negative log predictive density of validation set with rbf kernel of svd data %.3f" % neg_lpd_rbf_v)
	print("\n")

	print("End training model %d @ %.5f\n" % (j, time.time()-start))

	svd_train_acc[j] = accuracy_score(y_train, gp_rbf_fix.predict(X_train))
	print("Accuracy for X_train with rbf kernel of svd data: %.5f" % svd_train_acc[j])
	svd_test_acc[j] = accuracy_score(y_test, gp_rbf_fix.predict(X_test))
	print("Accuracy for X_test with rbf kernel of svd data: %.5f" % svd_test_acc[j])
	print("\n")

print("******************** final results using svd data ************************\n")
print("Average train accuracy of svd data: %.5f" % np.mean(svd_train_acc))
print("Average test accuracy of svd data: %.5f" % np.mean(svd_test_acc))
print("Average negative log predictive density of training set with rbf kernel of svd data: %.5f" % np.mean(nlpd_rbf_train_svd))
print("Average negative log predictive density of validation set with rbf kernel of svd data: %.5f" % np.mean(nlpd_rbf_test_svd))






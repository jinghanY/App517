from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import numpy as np
import os
def readData(fileName):
	f = open(fileName, "r")
	lines = f.readlines()
	header = lines[0]
	header_ele = header.split(";")
	header_ele = [x.strip('"') for x in header_ele]
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
	return data_feature

def loadLabels(inFileName):
	inFile = open(inFileName, 'r')
	lines = inFile.readlines()
	label = []
	for line in lines:
		label.append(int(float(line)))
	label = np.array(label)
	return label


def trainCla(features, label):
	model = LogisticRegression(penalty = 'l1')
	print("starting split dataset \n")
	X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.4)
	print("end for splitting dataset \n")

	print("************logistic regression with lasso**********")
	print("start training...\n")
	model.fit(X_train, y_train)
	print("end training.\n")
	# predict class labels for the train set
	print("start prediction for training data\n")
	predicted_train = model.predict(X_train)
	print("training accuracy")
	print(metrics.accuracy_score(y_train,predicted_train))




	# predict class labels for the test set
	print("start prediction for testing data \n")
	predicted = model.predict(X_test)
	print(predicted)
	# generate class probabilities
	probs = model.predict_proba(X_test)
	print(probs)
	print(np.shape(probs))
	print(len(probs[:,1]))
	# generate evaluation metrics
	print("testing accuracy")
	print(metrics.accuracy_score(y_test,predicted))
	print(metrics.roc_auc_score(y_test, probs[:,1]))
    # print(metrics.confusion_matrix(y_test, predicted))
    #
    # print(metrics.classification_report(y_test, predicted))

	print("************Decision Tree**********\n")
	print("start training for decision tree.\n")
	clf_tree = tree.DecisionTreeClassifier(max_depth=6)
	clf_tree = clf_tree.fit(X_train, y_train)
	
	print("training accuracy\n")
	predicted_tree_train = clf_tree.predict(X_train)
	print(metrics.accuracy_score(y_train,predicted_tree_train))


	print("test accuracy\n")	
	predicted_tree_test = clf_tree.predict(X_test)
	print(metrics.accuracy_score(y_test,predicted_tree_test))

	probs_tree = clf_tree.predict_proba(X_test)

	return probs[:,1], probs_tree[:,1], y_test

def diffThres(ypredProb, yReal):
	thresholds = np.linspace(0.0,1.0,num = 20)
	f1 = []
	for i in range(len(thresholds)):
		threshold_this = thresholds[i]
		print("len of ypredProb")
		print(len(ypredProb))
		yPred_this = getPred(ypredProb, threshold_this)
		print(yPred_this)
		print(yReal)
		print(type(yPred_this))
		print(len(yPred_this))
		print(type(yReal))
		print(len(yReal))
		f1_this = metrics.f1_score(yReal, yPred_this)
		f1.append(f1_this)
	print("thresholds")
	print(thresholds)
	print("f1 are")
	print(f1)
	hest_f1 = max(f1)
	print("highest f1 is")
	print(hest_f1)
	best_threshold = thresholds[f1.index(hest_f1)]

	maximum_call = sum(getPred(ypredProb, best_threshold))
	return hest_f1, best_threshold, maximum_call


def getPred(yPredProb, threshold):
	n = len(yPredProb)
	yPred = np.zeros(n)
	for i in range(n):
		if yPredProb[i] >= threshold:
			yPred[i] = int(1)
		else:
			yPred[i] = int(0)
	return yPred
		
	
	



if __name__ == "__main__":
	cwd = os.getcwd()
	file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
	file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"
	dataset = readData(file_name_feature)
	label = loadLabels(file_name_label)
	probs_log, probs_tree, y_test = trainCla(dataset, label)
	hest_f1, best_threshold, maximum_call = diffThres(probs_log, y_test)
	print("***************logistic regression******************\n")
	print("highest F1 score ")
	print(hest_f1)
	print("best threshold")
	print(best_threshold)
	print("maximum number to reach out")
	print(maximum_call)
	hest_f1_tree, best_threshold_tree, maximum_call_tree = diffThres(probs_tree, y_test)
	print("*************** Decision Tree ******************\n")
	print("highest F1 score ")
	print(hest_f1_tree)
	print("best threshold")
	print(best_threshold_tree)
	print("maximum number to reach out")
	print(maximum_call_tree)


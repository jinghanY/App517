import matplotlib.pyplot as plt
import pickle 
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


def trainCla(features, label, out_graph, header_ele,label_names):
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
	print(metrics.roc_auc_score(y_train,predicted_train))



	# predict class labels for the test set
	print("start prediction for testing data \n")
	predicted = model.predict(X_test)

	# generate class probabilities
	probs = model.predict_proba(X_test)

	# generate evaluation metrics
	print("testing accuracy")
	print(metrics.accuracy_score(y_test,predicted))
	# print(metrics.roc_auc_score(y_test, probs[:,1]))
	fpr_log,tpr_log,_= metrics.roc_curve(y_test, probs[:,1])
	roc_auc_log  = metrics.auc(fpr_log, tpr_log)
	print('AUC_test\n')
	print(roc_auc_log)
	
	# plot 
	plt.figure()
	lw = 2
	plt.plot(fpr_log,tpr_log, color = 'darkorange', lw=lw, label='ROC curve for class 1 (logistic regression)' )
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.savefig('logistic')


	#datapath = '/Users/jinghanyang/Dropbox/courses_WUSTL/2018spring/MachineLearning517/Application_project_own/App517/milestone_1'
	#fileName_probs = datapath+'/probs.pickle'
	#with open(fileName_probs,'wb') as f:
	#	pickle.dump(probs,f)
	#fileName_test = datapath + '/test.pickle'
	#with open(fileName_test, 'wb') as f:
	#	pickle.dump(y_test,f)
    # print(metrics.confusion_matrix(y_test, predicted))
    #
    # print(metrics.classification_report(y_test, predicted))
	#roc_auc =
	print()

	print("************Decision Tree**********\n")
	print("start training for decision tree.\n")
	clf_tree = tree.DecisionTreeClassifier(max_depth=6)
	clf_tree = clf_tree.fit(X_train, y_train)
	tree.export_graphviz(clf_tree, out_file = out_graph,feature_names=header_ele,class_names=label_names)
	
	print("training accuracy\n")
	predicted_tree_train = clf_tree.predict(X_train)
	print(metrics.accuracy_score(y_train,predicted_tree_train))


	print("test accuracy\n")	
	predicted_tree_test = clf_tree.predict(X_test)
	print(metrics.accuracy_score(y_test,predicted_tree_test))

	probs_tree = clf_tree.predict_proba(X_test)
	
	
	fpr_tree,tpr_tree,_= metrics.roc_curve(y_test, probs_tree[:,1])
	roc_auc_tree  = metrics.auc(fpr_tree, tpr_tree)
	print('AUC_test\n')
	print(roc_auc_tree)

	# plot 
	plt.figure()
	lw = 2
	plt.plot(fpr_tree,tpr_tree, color = 'darkorange', lw=lw, label='ROC curve for class 1 (tree)')
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.savefig('tree')


	return probs[:,1], probs_tree[:,1], y_test

def diffThres(ypredProb, yReal):
	thresholds = np.linspace(0.0,1.0,num = 20)
	f1 = []
	for i in range(len(thresholds)):
		threshold_this = thresholds[i]
		yPred_this = getPred(ypredProb, threshold_this)
		f1_this = metrics.f1_score(yReal, yPred_this)
		f1.append(f1_this)
	hest_f1 = max(f1)
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
	dataset,header_ele = readData(file_name_feature)
	label, class_names = loadLabels(file_name_label)
	probs_log, probs_tree, y_test = trainCla(dataset, label,"tree.dot", header_ele,class_names)
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


import numpy as np
import os

from sklearn.utils import shuffle
from scipy.stats import zscore
from sklearn.semi_supervised import label_propagation
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
import time

#readData() method converts the raw training data into computable data structure
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

#loadlabels() converts the raw training labels into computable data structure
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

#different_split() method conducts experiments of how to split the training set to determine the best split.
def different_split_experiment():
    best_split = 0
    best_acc = 0
    for i in range(1, 10): 
        X_labeled = training_dataset[:500 * i]
        y_labeled = training_labels_copy[0 : 500 * i]
        # X_unlabeled = training_dataset[500 * i : 5000]
        #y_unlabeled = training_labels_copy[500 * i : 5000]
        for j in range(500 * i, 5000):
            training_labels[j] = -1
        label_spread = label_propagation.LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.8)
        label_spread.fit(training_dataset, training_labels)
        y_training_predicted = label_spread.predict(X_labeled)
        y_testing_predicted = label_spread.predict(training_dataset)
        count = 0
        count1 = 0
        for k in range(1, 500* i):
            if(y_labeled[k] == y_training_predicted[k]):
                count = count + 1
        for l in range(1, 5000):
            if(training_labels_copy[l] == y_testing_predicted[l]):
                count1 = count1 + 1
        if best_acc < (count / (500 * i)):
            best_split = i
            best_acc = count / (500 * i)
        print("when the proportion of labeled data to the unlabeled data is " + str(i) + " : " + str((10 - i)) + ", the training accuracy of labeled training data is " + str(round(count / (500* i), 5)) + ", the testing acuracy of unseen data is " + str(round(count1 / (5000), 5)))
    print("the best split proportion of labeled data to unlabeled data is " + str(best_split) + " : " + str((10 - best_split)))
    return best_split

# RBF kernel parameter optimization
def RBFKernel_optimization_experiment():
    best_gamma = 0.0
    best_acc = -1
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    X_labeled = training_dataset[:500]
    y_labeled = training_labels_copy[0 : 500]
    for j in range(500 , 5000):
            training_labels[j] = -1
    for gamma in gammas:
        label_spread = label_propagation.LabelSpreading(kernel='rbf', gamma=gamma, alpha=0.8)
        label_spread.fit(training_dataset, training_labels)
        y_training_predicted = label_spread.predict(X_labeled)
        y_testing_predicted = label_spread.predict(training_dataset)
        count = 0
        count1 = 0
        for k in range(1, 500):
            if(y_labeled[k] == y_training_predicted[k]):
                count = count + 1
        for l in range(1, 5000):
            if(training_labels_copy[l] == y_testing_predicted[l]):
                count1 = count1 + 1
        if best_acc < (count1 / 5000):
            best_gamma = gamma
            best_acc = count1 / 5000
        print("when gamma is " + str(gamma) + ", the training accuracy is " + str(count/500) + ", the testing accuracy is " + str(count1/5000))
    print("the best gamma for RBF kernel is " + str(best_gamma))
    return best_gamma, best_acc

# KNN kernel parameter optimization
def KNNKernel_optimization_experiment():
    best_numNei = 0
    best_acc = -1
    numNeis = [1, 3, 5, 7, 10, 20, 40, 60, 80 ,100, 200, 400, 800, 1000]
    X_labeled = training_dataset[:500]
    y_labeled = training_labels_copy[0 : 500]
    for j in range(500 , 5000):
            training_labels[j] = -1
    for numNei in numNeis:
        label_spread = label_propagation.LabelSpreading(kernel='knn',n_neighbors=numNei, alpha=0.8)
        label_spread.fit(training_dataset, training_labels)
        y_training_predicted = label_spread.predict(X_labeled)
        y_testing_predicted = label_spread.predict(training_dataset)
        count = 0
        count1 = 0
        for k in range(1, 500):
            if(y_labeled[k] == y_training_predicted[k]):
                count = count + 1
        for l in range(1, 5000):
            if(training_labels_copy[l] == y_testing_predicted[l]):
                count1 = count1 + 1
        if best_acc < (count1 / 5000):
            best_numNei = numNei
            best_acc = count1 / 5000
        print("When the number of neighbor is " + str(numNei) + ", the training accuracy is " + str(count/500) + " the testing accuracy is " + str(count1/5000))
    print("the best number of neighbor is " + str(best_numNei))
    return best_numNei, best_acc

def alpha_optimization_experiment():
    best_acc = -1
    best_alpha = 0.0
    alphas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    X_labeled = training_dataset[:500]
    y_labeled = training_labels_copy[0 : 500]
    for j in range(500 , 5000):
            training_labels[j] = -1
    for alpha in alphas:
        label_spread = label_propagation.LabelSpreading(kernel='knn',n_neighbors=10, alpha=alpha)
        label_spread.fit(training_dataset, training_labels)
        y_training_predicted = label_spread.predict(X_labeled)
        y_testing_predicted = label_spread.predict(training_dataset)
        count = 0
        count1 = 0
        for k in range(1, 500):
            if(y_labeled[k] == y_training_predicted[k]):
                count = count + 1
        for l in range(1, 5000):
            if(training_labels_copy[l] == y_testing_predicted[l]):
                count1 = count1 + 1
        if best_acc < (count1 / 5000):
            best_alpha = alpha
            best_acc = count1 / 5000
        print("when alpha is " + str(alpha) + ", the training accuracy is " + str(count/500) + ", the testing accuracy is " + str(count1 / 5000))
    print("the best alpha is " + str(best_alpha))
    return best_alpha

    

#construct the training and testing dataset
cwd = os.getcwd()
file_name_feature = cwd + "/../dataset/bank-additional-full_new_features.csv"
file_name_label = cwd + "/../dataset/bank-additional-full_new_labels.csv"
dataset,header_ele = readData(file_name_feature)
labels, class_names = loadLabels(file_name_label)

training_dataset = shuffle(dataset, random_state=20)[:5000]
training_dataset[:, :9] = zscore(training_dataset[:, :9])
training_labels = shuffle(labels, random_state=20)[:5000]
training_labels_copy = shuffle(labels, random_state=20)[:5000]

testing_dataset = shuffle(dataset, random_state=20)[5001:10001]
testing_dataset[:, :9] = zscore(testing_dataset[:, :9])
testing_labels = shuffle(labels, random_state=20)[5001:10001]
# count0 = 0
# for label in labels:
#     if label == 0:
#         count0 = count0 + 1
# print("the number of 0 and the number of 1 in labels are respectively " + str(count0) + ", " + str(labels.shape[0] - count0))

#start the program
print("********the program start running**********\n")
print("*********split optimization**********\n")
N = different_split_experiment()
print("***************************************\n")
print("*********alpha optimization**********\n")
best_alpha = alpha_optimization_experiment()
print("***************************************\n")
print("*********kernel optimization**********\n")
RBF_gamma, RBF_acc = RBFKernel_optimization_experiment()
KNN_numNei, KNN_acc = KNNKernel_optimization_experiment()
print("***************************************\n")
# N=1
# best_alpha = 0.8
# RBF_gamma = 1
# KNN_numNei = 7
# kernel = 'knn'
# param = 20
elapsed = time.time()
print("*********10-Fold training based on RBF kernel**********\n")
training_acc = 0
testing_acc = 0
neg_log_train = 0
neg_log_test = 0
pointer = 0
pointer1 = 0

while pointer1 < 5000:
    X_labeled = training_dataset[pointer1 : pointer1 + (500 * N)]
    y_labeled = training_labels_copy[pointer1 : pointer1 + (500 * N)]
    # print(y_labeled)
    label_spread = label_propagation.LabelSpreading(kernel='rbf', gamma=RBF_gamma, alpha=best_alpha)
    label_spread.fit(training_dataset, training_labels)
    tmp_train= -np.mean(np.log(label_spread.predict_proba(X_labeled)[np.arange(len(X_labeled)), y_labeled]))
    tmp_test = -np.mean(np.log(label_spread.predict_proba(testing_dataset)[np.arange(len(testing_dataset)), testing_labels]))
    neg_log_train = neg_log_train + tmp_train
    neg_log_test = neg_log_test + tmp_test
    pointer1 = pointer1 + (500 * N)
neg_log_test = neg_log_test /10
neg_log_train = neg_log_train / 10


while pointer < 5000:
    X_labeled = training_dataset[pointer : pointer + (500 * N)]
    y_labeled = training_labels_copy[pointer : pointer + (500 * N)]
    # print(X_labeled[0])
    for h in range(1, 5000):
        if h <= pointer or h >= (pointer + (500 * N)):
            training_labels[h] = -1
    label_spread = label_propagation.LabelSpreading(kernel='rbf', gamma=RBF_gamma, alpha=best_alpha)
    label_spread.fit(training_dataset, training_labels)
    y_training_predicted = label_spread.predict(X_labeled)
    y_testing_predicted = label_spread.predict(testing_dataset)

    

    count = 0
    count1 = 0
    for k in range(1, 500 * N):
       if(y_labeled[k] == y_training_predicted[k]):
                count = count + 1
    for l in range(1, 5000):
        if(training_labels_copy[l] == y_testing_predicted[l]):
            count1 = count1 + 1
    training_acc = training_acc + (count / (500 * N) )
    testing_acc = testing_acc + (count1 / 5000)
    pointer = pointer + (500 * N)
training_acc = training_acc / 10
testing_acc = testing_acc / 10
print("The average training accuracy base on RBF kernel is " + str(training_acc) + ", and the average testing accuracy on unseen data is " + str(testing_acc))
print("Average negative log predictive density of training set with rbf kernel: %.5f" % neg_log_train)
print("Average negative log predictive density of training set with rbf kernel: %.5f" % neg_log_test)
new_time = time.time()
print("Total elapsed time: %.5f" % (new_time-elapsed))


print("*********10-Fold training based on KNN kernel**********\n")
training_acc = 0
testing_acc = 0
neg_log_train = 0
neg_log_test = 0
pointer = 0
pointer1 = 0

while pointer1 < 5000:
    X_labeled = training_dataset[pointer1 : pointer1 + (500 * N)]
    y_labeled = training_labels_copy[pointer1 : pointer1 + (500 * N)]
    label_spread = label_propagation.LabelSpreading(kernel='knn', n_neighbors=KNN_numNei, alpha=best_alpha)
    label_spread.fit(training_dataset, training_labels)
    tmp_train= -np.mean(np.log(label_spread.predict_proba(X_labeled)[np.arange(len(X_labeled)), y_labeled]))
    tmp_test = -np.mean(np.log(label_spread.predict_proba(testing_dataset)[np.arange(len(testing_dataset)), testing_labels]))
    neg_log_train = neg_log_train + tmp_train
    neg_log_test = neg_log_test + tmp_test
    pointer1 = pointer1 + (500 * N)
neg_log_test = neg_log_test /10
neg_log_train = neg_log_train / 10

    
while pointer < 5000:
    X_labeled = training_dataset[pointer : pointer + (500 * N)]
    y_labeled = training_labels_copy[pointer : pointer + (500 * N)]
    for h in range(1, 5000):
        if h <= pointer or h >= (pointer + (500 * N)):
            training_labels[h] = -1
    label_spread = label_propagation.LabelSpreading(kernel='knn', n_neighbors=KNN_numNei, alpha=best_alpha)
    label_spread.fit(training_dataset, training_labels)
    y_training_predicted = label_spread.predict(X_labeled)
    y_testing_predicted = label_spread.predict(testing_dataset)
    count = 0
    count1 = 0
    for k in range(1, 500 * N):
       if(y_labeled[k] == y_training_predicted[k]):
                count = count + 1
    for l in range(1, 5000):
        if(training_labels_copy[l] == y_testing_predicted[l]):
            count1 = count1 + 1
    training_acc = training_acc + (count / (500 * N) )
    testing_acc = testing_acc + (count1 / 5000)
    pointer = pointer + (500 * N)
training_acc = training_acc / 10
testing_acc = testing_acc / 10
print("The average training accuracy base on KNN kernel is " + str(training_acc) + ", and the average testing accuracy on unseen data is " + str(testing_acc))
print("Average negative log predictive density of training set with knn kernel: %.5f" % neg_log_train)
print("Average negative log predictive density of training set with knn kernel: %.5f" % neg_log_test)
new_time1 = time.time()
print("Total elapsed time: %.5f" % (new_time1-new_time))
    







    



import numpy as np
import re
import os
from oneHot import *
def preprocess(fileName, outFileName):
	f = open(fileName, "r")
	lines = f.readlines()
	header = lines[0]
	header_ele = header.split(";")
	header_ele = [x.strip('"') for x in header_ele]
	data_feature = []
	data_label = []
	ct = 0
	feature_cat = [2,3,4,5,6,7,8,9,10,15]
	for line in lines:
		ct = ct + 1
		if ct == 1:
			continue
		element = line.split(";")
		element_ele = [x.strip('"') for x in element]
		features = element_ele[0:-1]


		data_feature.append(features)
		label = element_ele[-1]
		label = re.sub(r'[^A-Za-z]', '', label)

		if label == "no":
			data_label.append(0)

		else:

			data_label.append(1)

	
	# convert dataset to numpy array
	data_feature = np.array(data_feature)
	# create a numpy array for numerical features
	# create a numpy array for categorical features
	num_datapoints,num_features = data_feature.shape
	num_cat_features = len(feature_cat)
	num_num_features = num_features - num_cat_features


	# dataset_num
	dataset_num = np.zeros((num_datapoints, num_num_features))
	header_cat = []
	header_num = []
	ct_num_fea = 0
	ct_cat = 0
	for i in range(num_features):
		if (i+1) in feature_cat:  # categorical feature 
			ct_cat = ct_cat + 1
			if ct_cat == 1:
				dataset_cat, header = oneHot(data_feature[:,i], num_datapoints)
				header_cat=header
			else:
				dataset_cat_this, header = oneHot(data_feature[:,i], num_datapoints)
				header_cat = header_cat + header

				dataset_cat = np.concatenate((dataset_cat,dataset_cat_this),axis=1)
		else:
			fea_num_this = (data_feature[:, i]).astype(np.float)

			dataset_num[:,ct_num_fea] = fea_num_this
			ct_num_fea = ct_num_fea +1
			header_num.append(header_ele[i])
	
	header_new = header_num + header_cat
	dataset_new = np.concatenate((dataset_num, dataset_cat), axis=1)

	return dataset_new, header_new, data_label

def fileWrite_features(dataset, header,outFileName):
	outFile = open(outFileName, 'w')
	for word in header:
		outFile.write(word+';')
	outFile.write('\n')
	for row in range(np.shape(dataset)[0]):
		for col in range(np.shape(dataset)[1]):
			outFile.write(str(dataset[row, col]) + ';')
		outFile.write('\n')
	outFile.close()
 
 
def fileWrite_label(labels, outFileName):
    outFile = open(outFileName, 'w')
    for label in labels:
        outFile.write(str(label)+'\n')
    outFile.close()



if __name__ == "__main__":
	cwd = os.getcwd()
	file_name = cwd + "/../dataset/bank-additional-full.csv"

	outFile_name_features = cwd + "/../dataset/bank-additional-full_new_features.csv"
	dataset_new, header_new, data_label = preprocess(file_name, outFile_name_features)
	outFile_name_labels = cwd + "/../dataset/bank-additional-full_new_labels.csv" 
	fileWrite_label(data_label, outFile_name_labels)
	fileWrite_features(dataset_new, header_new,outFile_name_features)

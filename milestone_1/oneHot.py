import numpy as np
# input is a categorical feature, it's an array, n is the number of data points. m is the total number of existing features..
# output is a number of features after one-hot encoding and number of new features.output is a numpy array. 
def oneHot(aFea, n):
	# create dictionary
	dict = {}
	header = []
	m = 0
	for i in range(n):
		if aFea[i] not in dict:
			dict[aFea[i]] = m   # value is one-hot feature number, key is the feature name. (the original string name.)
			m = m + 1
			header.append(aFea[i])

	
	# generate one hot features 
	numNewFea = len(dict.keys())  # total number of new features for this category is number of keys in the dictionary.
	# create a numpy array		
	dataset_clip = np.zeros((n,numNewFea))

	for i in range(n):
		fea_this = aFea[i]
		dataset_clip[i, dict[fea_this]]	= 1

	return dataset_clip, header
			
	
	

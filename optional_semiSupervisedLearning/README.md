# Optional - Semi-Supervised-Learning

## Highlights
- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- Parameter Optimization 
- RBF Kernel
- KNN Kernel
- 10-Fold Cross validation


## Dataset
We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset has 41188 customers and 20 features (63 features after applying one-hot encoding). Since the output variable is a binary number, this is a binary classification problem. For this Task, We chose 5000 data points to construct dataset, since the point of semi-supervised learning is to take advantage of the data points whose labels are unclear, we need to set some labels of the data points to be -1 (due to the requirement of Sklearn.semi_supervised.LabelSpreading). Then we contructed the trainingdataset. In order to improve the reliablity of the result, we selected another 5000 unseen datapoints to serve as a 'testing dataset'. Meanwhile, we used z-score for 10 numerical features for normalization and applied one-hot encoding for 10 categorical features.

## Dataset Split
Since the Semi-Supervised Learning use unlabeled data points to gain information about the model or boundary, we need to determine the best fraction to split the dataset. We divided the dataset of 5000 data points into 10 parts, with 500 data points each, then we tried different fraction of number of labeled data points to number of unlabeled ones, such as 1:9, 2:8, 3:7 and so on. After a lot of experiments, we found out the best split strategy is the fraction of labeled to unlabeled is 1 : 9.


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
We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset has 41188 customers and 20 features (63 features after applying one-hot encoding). Since the output variable is a binary number, this is a binary classification problem. For this Task, We chose 5000 data points to construct dataset, since the point of semi-supervised learning is to take advantage of the data points whose labels are unclear, we need to set some labels of the data points to be -1 (due to the requirement of Sklearn.semi_supervised.LabelSpreading). Then we have contructed the training dataset. In order to improve the reliablity of the result, we selected another 5000 unseen datapoints to serve as a 'testing dataset'. Meanwhile, we used z-score for 10 numerical features for normalization and applied one-hot encoding for 10 categorical features.

## SKLearn Package
For this task, we used the sklearn.semi_supervised package. scikit-learn provides two label propagation models: LabelPropagation and LabelSpreading. Both work by constructing a similarity graph over all items in the input dataset. LabelPropagation uses the raw similarity matrix constructed from the data with no modifications. In contrast, LabelSpreading minimizes a loss function that has regularization properties, as such it is often more robust to noise. Based on the comparsion, we used LabelSpreading model to train our dataset

## Dataset Split
Since the Semi-Supervised Learning use unlabeled data points to gain information about the model or boundary, we need to determine the best fraction to split the dataset. We divided the dataset of 5000 data points into 10 parts, with 500 data points each, then we tried different fraction of number of labeled data points to number of unlabeled ones, such as 1:9, 2:8, 3:7 and so on. After a lot of experiments, we found out the best split strategy is the fraction of labeled to unlabeled is 1 : 9.

## Parameter Alpha Optimization
The clamping factor Alpha whose value is in [0, 1], specifies the relative amount that an instance should adopt the information from its neighbors as opposed to its initial label. alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information. We conducted experiments to iterate through 0.01, 0.1, 0.2, 0.3, 0.4, 0.5,...,0.99 to determine which alpha is the most suitable value for this dataset. We concluded that the testing accuracy is the highest when the alpha is 0.1.

## Kernel Parameter Optimization
The LabelSpreading model has two kernels to choose from, RBF kernel and KNN kernel. For this task we optimiz both parameters of these two kernels. For the gamma of RBF kernel, we iterate through 0.001, 0.01, 0.1, 1, 10, 100, 1000 to track the best gamma parameter for training the model. After experiments, we found out the best gamma for RBF kernel is 1. For its counterpart, we chose 1, 3, 5, 7, 10, 20, 40, 60, 80 ,100, 200, 400, 800, 1000 to figure out which one is the best parameter for KNN kernel. We conducted experiements and it turned out to be the 7 is the best number of neighbor for KNN kernel and 1 to be the best gamma value for RBF kernel.

## Cross Validation
Since the speciality of the unlabeled data points, we coded our own 10-fold cross validation. Which is split the dataset into 10 parts equally, after we decide how to split the dataset, then each time selected the best number of data points to be the labeled data and set other labels to -1, after the fit and predict, switch to the next same amount of data points and do the same process. After went through all data points, averaged the accuracy each iteration got to get the average training and testing accuracy.

## Error measurement
We used accuracy and negative log predictive density as error measurement. The accuracy is simply the fraction of true predictions among training/validation set, which gives us a roughly evaluation of the model. The negative log predictive density, can evaluate the model more precise since it takes the predictive probability into account. The expression for the NLPD is as follows.

![NLPD expression](https://latex.codecogs.com/gif.latex?L=-\frac{1}{n}\sum^{n}_{i=1}\log{p(y_i=t_i|\mathbf{x_i})})

Note that the log function is monotonically increasing, and the greater the probability is, the better the model is. Therefore, the smaller (closer to 0) the NLPD is, the better the model is.

## Result
Here are the final results of semi-supervised learning based on different kernel function.
```
**************** final results with RBF kernel ********************

Average training accuracy: 0.89662
Average testing accuracy: 0.88825
Average negative log predictive density of training set with rbf kernel: 0.51965
Average negative log predictive density of validation set with rbf kernel of raw data: 0.4853
Running time for 10-fold cross validation: 34.29782

******************** final results with KNN kernel ************************

Average training accuracy: 0.89499
Average testing accuracy: 0.88172
Average negative log predictive density of training set with rbf kernel: 0.18492
Average negative log predictive density of validation set with rbf kernel of raw data: 0.23590
Running time for 10-fold cross validation: 18.23773
```
We can see the two models above have comparable average performance, no mater the training accuracy or the testing one. But when we compare the negative log predictive density of these two model, we can find out the one based on the KNN kernel is better than its counterpart. Meanwhile, the running time of these two model agree on the previous statement, which is the KNN based model is faster and performs better then the model base on RBF. 





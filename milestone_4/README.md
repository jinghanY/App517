# Milestone 4 - Final Comparison

## Highlights
- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- RBF Kernel, KNN Kernel, Matern Kernel
- 10-Fold Cross validation
- Support Vector Machine
- Decision Tree
- Logistic Regression
- Gaussian Process
- Semi-Supervised Learning


## Dataset
We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset has 41188 customers and 20 features (63 features after applying one-hot encoding). Since the output variable is a binary number, this is a binary classification problem. For this Task, We chose 5000 data points to construct dataset, During the comparsion, we keep the dataset consistent to improve the reliablity of the results. Meanwhile, we used z-score for 10 numerical features for normalization and applied one-hot encoding for 10 categorical features.

## Cross Validation
Since the speciality of semi-supervised learning, we coded our own 10-fold cross validation for this model, which split the dataset into 10 parts equally, after we decide how to split the dataset, each time we selected the best number of data points to be the labeled data and set other labels to -1, after the fit and predict, switch to the next same amount of data points and do the same process. After went through all data points, averaged the accuracy each iteration got to get the average training and testing accuracy.  For the rest of other models, we used sklearn.model_selection.StratifiedKFold to process the training/validation split. The advantage of this method is that it preserves the percentage of samples for each class.

## Error measurement
We used accuracy and negative log predictive density as error measurement. The accuracy is simply the fraction of true predictions among training/validation set, which gives us a roughly evaluation of the model. The negative log predictive density, can evaluate the model more precise since it takes the predictive probability into account. The expression for the NLPD is as follows.

![NLPD expression](https://latex.codecogs.com/gif.latex?L=-\frac{1}{n}\sum^{n}_{i=1}\log{p(y_i=t_i|\mathbf{x_i})})

Note that the log function is monotonically increasing, and the greater the probability is, the better the model is. Therefore, the smaller (closer to 0) the NLPD is, the better the model is.

## Result
Here are the final results of semi-supervised learning based on different kernel function.
```
**************** final results with RBF kernel ********************

|  Model Name   | Training Acc | Testing Acc | Training NLPD | Testing NLPD | Running TimeD |  
|---------------|--------------|-------------|---------------|--------------|---------------|
|     SVM       |   0.94733    |   0.90840   |    0.12958    |   0.23351    |   68.71207    |
| Decision Tree |   0.93616    |   0.95208   |    0.14917    |   0.11697    |   1.87422     |
|   Logistic    |   0.91668    |   0.91001   |    0.20387    |   0.21263    |   3.04642     |
|   RBF_GP      |   0.91553    |   0.91021   |    0.19745    |   0.21109    |   445.20951   |
|  MATERN_GP    |   0.93907    |   0.90960   |    0.15003    |   0.19460    |   907.11759   |
|   SVD_GP      |   0.90169    |   0.90120   |    0.27720    |   0.27867    |   429.92041   |
|   RBF_SSL     |   0.89660    |   0.89660   |    0.51965    |   0.48530    |   34.29781    |
|   KNN_SSL     |   0.89499    |   0.88172   |    0.18492    |   0.23590    |   18.23773    |

Training Acc = Average Training Accuracy
Testing Acc = Average Testing Accuracy
Training NLP = Average Negative Log Predictive Density of training set
Testing NLP = Average Negative Log Predictive Density of testing set
SVM = Support Vector Machine
Logistic = Logistic Regression
RBF_GP = Gaussian Process with RBF Kernel
MATERN_GP = Gaussian Process with MATERN Kernel
SVD_GP = Singular Value Decomposition based Gaussian Process
RBF_SSL = Semi-Supervised Learning with RBF kernel
KNN_SSL = Semi-Supervised Learning with KNN kernel

```


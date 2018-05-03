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
**************** Results of comparsion between different models ********************

|  Model Name   | Training Acc | Testing Acc | Training NLPD | Testing NLPD | Running Time  |  
|---------------|--------------|-------------|---------------|--------------|---------------|
|     SVM       |   0.94733    |   0.90840   |    0.12958    |   0.23351    |   68.71207    |
| Decision Tree |   0.93616    |   0.95208   |    0.14917    |   0.11697    |   1.87422     |
|   Logistic    |   0.91668    |   0.91001   |    0.20387    |   0.21263    |   3.04642     |
|   RBF_GP      |   0.91553    |   0.91021   |    0.19745    |   0.21109    |   445.20951   |
|  MATERN_GP    |   0.93907    |   0.90960   |    0.15003    |   0.19460    |   907.11759   |
|   SVD_GP      |   0.90169    |   0.90120   |    0.27720    |   0.27867    |   429.92041   |
|   RBF_SSL     |   0.89660    |   0.89660   |    0.51965    |   0.48530    |   34.29781    |
|   KNN_SSL     |   0.89499    |   0.89499   |    0.18492    |   0.23590    |   18.23773    |

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
The table above shows some results of the methods that we have implemented for this semester, which includes SVM, Logistic Regression, Gaussian Process and Semi_Supervised Learning. As we know, the most significant factor of determining whether a model is good or not is the accuracy. We used 10-fold cross validation to get the average accuracy of 10 training. From this perspective, we can see the SVM model has the highest training accuracy and the Semi-Supervised Learning with KNN kernel has the lowest traning accuracy. For the comparsion of the testing accuracy of these models, the Decision Tree model has the highest testing accuracy, but the testing accuracy is higher than the training accuracy, which indicate that the desicion tree has some limitation to learn from the data because of the data structure. Then the Gaussian Process with RBF kernel has the second highest testing accuracy. 

Usually we will use the testing accuracy to be the most reliable standard to determine if the model is good or bad. As we discussed above, the Gaussian Process has the best achievable average testing accuracy, but the running time for 5000 data points is around 450 seconds, which is much longer than the model with the third highest average accuracy: Logistic Regession. When we look into the Average Negative Log Predictive Density of training set and testing set, the values of Logistic Regession are also acceptable. Then we can conclude that for the dataset with the purpose of selling a financial product to potential customers, the logistic regression is the most suitable model for classification, which achieved 0.91001 predicted accuracy in about 5 seconds.

For the model with the worst performance, we tried to look deeper into the data structure of the given dataset, found out that the labels of datasets is unbalanced, The number of 0 is 36548, which is about 88.73% of all the labels. Because of this data structure, the model predict mistakenly by taking advantage of the information retrieved from the unlabeled data. 




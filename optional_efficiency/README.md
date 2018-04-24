# Optional - Efficiency

## Highlights
- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- 4 methods of classification: logistic regression, Gaussian Process, Support Vector Machine and Gaussian Process after SVD
- Cross validation with StratifiedKFold, which preserves the percentage of samples for each class
- Sign test on the training times and testing times to determine whether one method is more efficiency than another


## Dataset
We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset has 41188 customers and 20 features (63 features after applying one-hot encoding). Since the output variable is a binary number, this is a binary classification problem. Since much data points will increase the running time of Gaussian Process greatly, we choose 5000 data points randomly from the dataset to train and test the 4 classification methods.
We used z-score for 10 numerical features for normalization and applied one-hot encoding for 10 categorical features.

## Four methods that in the comparison
For this task, the 4 classification methods we used and their parameters are described below:

1. The first method is logistic regression with the l1 regularization.

2. The second method is Gaussian Process using a RBF kernal, which is 76.5**2 * RBF(length_scale=179).

3. The third method is Support Vector Machine using a RBF kernal, which set gamma = 0.1 and C = 1. The gamma parameter defines how far the influence of a single training example reaches. The C parameter trades off misclassification of training examples against simplicity of the decision surface.

4. The fourth method is Gaussian Process using a RBF kernal that is same as the second method. The difference from the second method is that it uses the dataset after SVD.

## Cross-validation
We used sklearn.model_selection.StratifiedKFold to process the training/validation split. The advantage of this method is that it preserves the percentage of samples for each class. Then we used these training and testing set to run the 4 classification methods and record the training and testing time of every method. For each two methods, we applied a statistical test on their training and testing time to determine whose efficiency is better, which we will discuss in the next section.

## Statistical test
For every method, we apply a 10-fold cross validation which means we had 10 training times and testing times for every method. We printed the average training and testing time for every method, but we want to further find a statistical test to help us determine the efficiency of every method. We decided to use sign test.

For this task, it is a sign test for two classifiers on one domain. When comparing two classifiers A and B, we let Awin to be the number of A outperforms B and Bwin to be the number of B outperforms A. Our null hypothesis is that the probability that A outperforms B is 0.5. In this case, Awin is the times that the training/testing time of A is smaller than the training/testing time of B. We picked the significant level of the sign test, which is alpha, to be 0.05, which means for 10 running times, Awin should larger than Walpha = 8 to be consider significant better than B on efficiency, Bwin should also larger than Walpha = 8 to be consider significant better than A on efficiency. If both Awin and Bwin are smaller than Walpha = 8, then we consider A and B have similar efficiency.

## Result
Here are the average training and testing time of 4 models,
```
************* final results for the efficiency of linear classification using raw data ****************

Average training time of linear calssification: 0.19854
Average testing time of linear calssification: 0.00013

************* final results for the efficiency of GP classification using raw data ****************

Average training time of GP calssification: 24.85182
Average testing time of GP calssification: 0.19172

************* final results for the efficiency of SVM classification using raw data ****************

Average training time of SVM calssification: 0.62919
Average testing time of SVM calssification: 0.06436

************* final results for the efficiency of GP classification using SVD data ****************

Average training time of SVD GP calssification: 24.05429
Average testing time of SVD GP calssification: 0.06389
```

Here are the final results about the comparison of 4 models.
```
*************** Comparing between  linear and  GP  *****************

linear  is better than  GP  w.r.t efficiency of training

linear  is better than  GP  w.r.t efficiency of testing

*************** Comparing between  linear and  SVM  *****************

linear  is better than  SVM  w.r.t efficiency of training

linear  is better than  SVM  w.r.t efficiency of testing

*************** Comparing between  linear and  SVD_GP  *****************

linear  is better than  SVD_GP  w.r.t efficiency of training

linear  is better than  SVD_GP  w.r.t efficiency of testing

*************** Comparing between  GP and  SVM  *****************

SVM  is better than  GP  w.r.t efficiency of training

SVM  is better than  GP  w.r.t efficiency of testing

*************** Comparing between  GP and  SVD_GP  *****************

GP  and  SVD_GP  is similar w.r.t efficiency of training

SVD_GP  is better than  GP  w.r.t efficiency of testing

*************** Comparing between  SVM and  SVD_GP  *****************

SVM  is better than  SVD_GP  w.r.t efficiency of training

SVM  and  SVD_GP  is similar w.r.t efficiency of testing
```
From the result, we can see that for training, the efficiency rank for the 4 methods is linear > SVM > GP = GP_SVD. And for testing, the rank is linear > SVM = GP_SVD > GP. We can have the conclusion that SVD on the raw dataset can improve the efficiency of that method, but the main reason for efficiency is the classification method not the dataset.

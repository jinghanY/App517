# Milestone 3 - Dimensionality Reduction

## Highlights
- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- Dimensionality reduction to dataset using SVD
- Cross validation with StratifiedKFold, which preserves the percentage of samples for each class
- Accuracy and negative log predictive density as measurement


## Dataset
We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset has 41188 customers and 20 features (63 features after applying one-hot encoding). Since the output variable is a binary number, this is a binary classification problem. In order to highlight the differences between two Gaussian Process methods using dataset with and without SVD and also decrease the running time, we choose 5000 data points randomly from the dataset to train and test the Gaussian Process classification.
We used z-score for 10 numerical features for normalization and applied one-hot encoding for 10 categorical features.

## Dimensionality reduction to dataset and visualization
For this task, we used numpy.linalg.svd to apply Singular Value Decomposition to our raw dataset, which can be described as:

![SVD expression](https://latex.codecogs.com/gif.latex?X&space;=&space;USV^{T})

Where X is our raw dataset, U contains the left singular vectors, Si are the singular values of X and V contains the right singular vectors.

*One thing need to keep in mind is that the S returned by numpy.linalg.svd is an array of singular values of X which have been sorted from largest singular value to the smallest singular value. Also the order of left singular vectors in U returned by numpy.linalg.svd is the same as the order of S, so if we want to pick n singular vectors with the largest singular values, we only need to pick the first n columns of U matirx.*

Since we wanted to visualize the dataset after SVD, we chose to remain the first and second principal component, which is, we only picked the first 2 columns of U matrix to reconstruct the data matrix using this equation:

![Reconstruction expression](https://latex.codecogs.com/gif.latex?\dot{X}&space;=&space;U^{T}X)

After reconstruction of the dataset, we used matplotlib.pyplot.plot to visualize the data matrix, where blue 'x' represents data points with label 0 and red '+' represents data points with label 1. You can find the figure of the dataset in the folder milestone_3.

## Cross-validation
We used sklearn.model_selection.StratifiedKFold to process the training/validation split. The advantage of this method is that it preserves the percentage of samples for each class. Then we use the Gaussian Process with RBF kernal to train and test the model using dataset with and without SVD. For each of the models, we evaluate it with certain error measurement, which we will discuss in the next section.

## Error measurement
We used accuracy and negative log predictive density as error measurement. The accuracy is simply the fraction of true predictions among training/validation set, which gives us a roughly evaluation of the model. The negative log predictive density, can evaluate the model more precise since it takes the predictive probability into account. The expression for the NLPD is as follows.

![NLPD expression](https://latex.codecogs.com/gif.latex?L=-\frac{1}{n}\sum^{n}_{i=1}\log{p(y_i=t_i|\mathbf{x_i})})

Note that the log function is monotonically increasing, and the greater the probability is, the better the model is. Therefore, the smaller (closer to 0) the NLPD is, the better the model is.

## Result
Here are the final results about the models using dataset with and without SVD.
```
**************** final results using raw data ********************

Average train accuracy of raw data: 0.91556
Average test accuracy of raw data: 0.91021
Average negative log predictive density of training set with rbf kernel of raw data: 0.19745
Average negative log predictive density of validation set with rbf kernel of raw data: 0.21109

******************** final results using svd data ************************

Average train accuracy of svd data: 0.90193
Average test accuracy of svd data: 0.90120
Average negative log predictive density of training set with rbf kernel of svd data: 0.27720
Average negative log predictive density of validation set with rbf kernel of svd data: 0.27867
```
From the result, we can see that GP with raw data, the average accuracy is higher and the NLPD of the training and testing sets is lower than GP with data after SVD. Since SVD will reduce the dimension of dataset, it is sure that we will lose some of the imformation of the dataset after SVD. This is why that the accuracy and NLPD for SVD dataset are all worse than raw dataset. But the main advantage of SVD is that it will decrease the running time of the classification, which we will describe in the optional task ---- efficiency.

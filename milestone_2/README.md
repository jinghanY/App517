# Milestone 2 - Gaussian Process

## Highlights
- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- Cross validation with StratifiedKFold, which preserves the percentage of samples for each class
- RBF Kernel
- Matern Kernel
- Accuracy and negative log predictive density as measurement


## Dataset
We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset has 41188 customers and 20 features (63 features after applying one-hot encoding). Since the output variable is a binary number, this is a binary classification problem. Considering that a dataset of this size would take a long time for Gaussian Process, we chose 1000 data points randomly as the dataset to run GP.
We used z-score for 10 numerical features for normalization and applied one-hot encoding for 10 categorical features.

## Kernel selection and parameter optimization
For this task, we used two different kernels: RBF kernel and Matern kernel. Prior to train the model, we first used the built-in optimizer for GaussianProcessClassifier to find the best parameters for each of the kernel. The results are as follows.
```
Best kernel is  76.5**2 * RBF(length_scale=179)
Best kernel is  3.7**2 * Matern(length_scale=9.4, nu=1.5)
```
About Matern kernels, here are some brief information about it.

![Matern expression](http://scikit-learn.org/stable/_images/math/3073f503e2051eae1f6768f094d9e8d3ebe3ae3d.png)

*The class of Matern kernels is a generalization of the RBF and the absolute exponential kernel parameterized by an additional parameter nu. The smaller nu, the less smooth the approximated function is. For nu=inf, the kernel becomes equivalent to the RBF kernel and for nu=0.5 to the absolute exponential kernel. Important intermediate values are nu=1.5 (once differentiable functions) and nu=2.5 (twice differentiable functions).*

## Cross-validation
We used sklearn.model_selection.StratifiedKFold to process the training/validation split. The advantage of this method is that it preserves the percentage of samples for each class. Then we use the kernels along with the parameters that we previously acquired to train the model. For each of the models, we evaluate it with certain error measurement, which we will discuss in the next section.

## Error measurement
We used accuracy and negative log predictive density as error measurement. The accuracy is simply the fraction of true predictions among training/validation set, which gives us a roughly evaluation of the model. The negative log predictive density, can evaluate the model more precise since it takes the predictive probability into account. The expression for the NLPD is as follows.

![NLPD expression](https://latex.codecogs.com/gif.latex?L=-\frac{1}{n}\sum^{n}_{i=1}\log{p(y_i=t_i|\mathbf{x_i})})

Note that the log function is monotonically increasing, and the greater the probability is, the better the model is. Therefore, the smaller (closer to 0) the NLPD is, the better the model is.

## Result
Here are the last few lines of the result.
```
Average accuracy with rbf kernel: 0.94200
Average accuracy with matern kernel: 0.95755
Average negative log predictive density of training set with rbf kernel: 0.15083
Average negative log predictive density of training set with matern kernel: 0.14716
Average negative log predictive density of validation set with rbf kernel: 0.19793
Average negative log predictive density of validation set with matern kernel: 0.22344
Total elapsed time: 9.88400
```
From the result, we can see that with matern kernel, the average accuracy is higher and the NLPD of the training set is lower. However, the NLPD is of the validation set of matern kernel is higher of that with RBF kernel, which may indicate some overfitting in the model with matern kernel.

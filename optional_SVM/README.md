# Optional - Support Vector Machine

## Hightlight
- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- Cross validation with StratifiedKFold, which preserves the percentage of samples for each class
- RBF kernel with parameters determined by grid search
- Accuracy, confusion matrix, and F1-score as measurement

## Dataset
We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset has 41188 customers and 20 features (63 features after applying one-hot encoding). Since the output variable is a binary number, this is a binary classification problem. Considering that a dataset of this size would take a long time for kernel SVM, we chose 5000 data points randomly as the dataset to train the model.
We used z-score for 10 numerical features for normalization and applied one-hot encoding for 10 categorical features.

## Kernel selection and parameter optimization
For this task, we used two different kernels: RBF kernel. Different from what we did for parameter optimizing the Gaussian Process, we used grid search to find the best parameters gamma and C for the RBF kernel. The result are as follows.
```
The best parameters are {'gamma': 0.10000000000000001, 'C': 1.0} with a score of 0.90
```
About the parameters of the RBF kernel, here are some brief information.
The gamma parameter defines how far the influence of a single training example reaches, with low values meaning ¡®far¡¯ and high value meaning ¡®close¡¯. While the C parameter trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors.

## Cross-validation
We used sklearn.model_selection.StratifiedKFold to process the training/validation split. The advantage of this method is that it preserves the percentage of samples for each class. Then we use the kernels along with the parameters that we previously acquired to train the model. For each of the models, we evaluate it with certain error measurement, which we will discuss in the next section.

## Error measurement
We used accuracy, confusion matrix, and F1-score as error measurement. The accuracy is simply the fraction of true predictions among training/validation set, which gives us a roughly evaluation of the model. Since 9/10 of the data are classified as ¡®0¡¯, we used the confusion matrix to check its TNR (True Positive Rate) and FNR (False Negative Rate). We also used F1-score to measure the accuracy. The expression for F1-score is as follows. 

![F1-score](https://latex.codecogs.com/gif.latex?F_1=\frac{2}{\frac{1}{\text{recall}}+\frac{1}{\text{precision}}}=2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}})

## Result
Here are the results of one of the models with RBF kernel.
```
TNR, FNR = 0.99524, 0.40316
F1-score: 0.73035
Accuracy for X_train: 0.95046
Accuracy for X_test: 0.90180
```
And the average accuracy for all validation set is,
```
Average accuracy for X_test 0.90279
```
We can see that the TNR is almost 1, but the FNR is also a huge number; the F1-score¡¯s fair; the accuracy for X_train is high, but the accuracy for X_test is only 2% higher than considering all samples as '0'.
Then I tried with a dataset that contains half labeled as '0' and half labeled as '1', and here are the results.
```
TNR, FNR = 0.87022, 0.05244
F1-score: 0.91228
Accuracy for X_train: 0.90889
Accuracy for X_test: 0.88200

Average accuracy for X_test 0.86900
```
The FNR is much lower than before, but the accuracy is also lower. The F1-score is higher.

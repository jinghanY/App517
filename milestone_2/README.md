# Optional - Support Vector Machine

We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset have 41188 customers and 20 features (63 features after applying one-hot encoding). Considering that the size of the dataset is too big for Gaussian Process, we chose 5000 data points randomly as the dataset to run SVC.

- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- Cross validation with StratifiedKFold, which preserves the percentage of samples for each class
- RBF kernel with C = 1 and gamma = 0.1 (parameters are determined by searching)
- Linear kernel
- Accuracy, confusion matrix, and F1-score as measurement
* F1-score

![F1-score]https://latex.codecogs.com/gif.latex?F_1=\frac{2}{\frac{1}{\text{recall}}+\frac{1}{\text{precision}}}=2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}

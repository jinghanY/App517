# Milestone 2 - Gaussian Process

We kept using the dataset with the purpose of selling a financial product to potential customers. The original dataset have 41188 customers and 20 features (63 features after applying one-hot encoding). Considering that the size of the dataset is too big for Gaussian Process, we chose 1000 data points randomly as the dataset to run GP.

- Binary classification
- One-hot encoding for 10 categorical features
- z-score for 10 numerical features
- Cross validation with StratifiedKFold, which preserves the percentage of samples for each class
- rbf Kernel: 3.49 ** 2 * RBF(length_scale=5.28)
- Matern Kernel: 3.84 ** 2 * Matern(length_scale=8.06, nu=1.5)
- Accuracy and negative log predictive density as measurement

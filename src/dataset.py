import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import root_mean_squared_error, r2_score

class NeuralDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# Actually we should implement a Softmax or use other classification model in the future implementation.
class LinearRegressionModel:
    def __init__(self, input_dim, output_dim):
        self.model = [Lasso(alpha=0.1) for _ in range(output_dim)]# [LinearRegression() for _ in range(output_dim)]
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def learn(self, X, y):
        for i in range(self.output_dim):
            self.model[i].fit(X, y[:, i])
    
    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.output_dim))
        for i in range(self.output_dim):
            y_pred[:, i] = self.model[i].predict(X)
        return y_pred
    
    def evaluate(self, X, y, mode = 'R2'):
        '''mode should be 'R2' or 'RMSE', or SNR(signal - noise Ratio = $$-10 log_10 (1-R^2))$$'''
        y_pred = self.predict(X)
        if mode == 'R2':
            return r2_score(y, y_pred)
        elif mode == 'RMSE':
            return root_mean_squared_error(y, y_pred)
        elif mode == 'SNR':
            return -10 * np.log10(1 - r2_score(y, y_pred))
import numpy as np

class LogisticRegression:
    def __init__(self, data):
        self.instances = data
        self.m, self.n = len(data), len(data[0])-1
        self.theta = np.zeros((self.n + 1, 1))
        self.train_x = np.array([[1] + i[:-1] for i in self.instances], dtype=np.float64)
        self.train_y = np.array([i[-1] for i in self.instances], dtype=np.float64)
        self.train_y = self.train_y.reshape((self.m, 1))
        self.train()
    
    def train(self, alpha = 0.01, max_iter = 1000):
        self.gradient_descent(alpha, max_iter)

    def gradient(self):
        y_hat = self.sigmoid(np.dot(self.train_x, self.theta))
        grad = (1/ self.m) * np.dot(self.train_x.T, (y_hat - self.train_y))
        return grad

    def gradient_descent(self, alpha = 0.01, max_iter = 1000):
        for _ in range(max_iter):
            self.theta = self.theta - alpha * self.gradient()

    def evaluate(self, x, prob = False, both = False):
        x = np.array(x, dtype=np.float64)
        
        y_pred = self.sigmoid(np.dot(x, self.theta))
        y_maj = 1 if y_pred >= 0.5 else 0

        if both: return y_maj, y_pred
        if prob: return y_pred
        return y_maj

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
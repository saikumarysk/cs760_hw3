import numpy as np

def find_mean(x):
    return np.mean(x, axis=0).reshape((1, x.shape[1]))

def find_std(x):
    return np.std(x, axis=0).reshape((1, x.shape[1]))

class LogisticRegression:
    def __init__(self, data):
        self.instances = data
        self.m, self.n = len(data), len(data[0])-1
        self.theta = np.zeros((self.n + 1, 1))
        self.train_x = np.array([[1] + i[:-1] for i in self.instances], dtype=np.float64)
        self.train_y = np.array([i[-1] for i in self.instances], dtype=np.float64)
        self.train_y = self.train_y.reshape((self.m, 1))

        self.mu = find_mean(self.train_x)
        self.sigma = find_std(self.train_x)
        self.sigma[self.sigma == 0] = 1
        self.train_x = (self.train_x - np.repeat(self.mu, repeats=self.m, axis=0)) / self.sigma
        #for _ in range(self.d):
        #    self.train_x[:, _] = self.train_x[:, _] / self.sigma[_]
        
        #self.mu = self.mu.reshape((self.d, 1))
        #self.sigma = self.sigma.reshape((self.d, 1))

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
        x = (x - self.mu) / self.sigma
        
        y_pred = self.sigmoid(np.dot(x, self.theta))
        y_maj = 1 if y_pred >= 0.5 else 0

        if both: return y_maj, y_pred
        if prob: return y_pred
        return y_maj

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
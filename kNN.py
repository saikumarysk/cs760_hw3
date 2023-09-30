import numpy as np

class kNN:
    def __init__(self, k, data):
        self.k = k
        self.instances = data
        self.m = len(data)
        self.d = len(data[0]) - 1
        self.instances = np.array(self.instances, dtype=np.float64)
    
    def evaluate(self, x, prob = False, both = False):
        x = np.array(x, dtype=np.float64).reshape((self.d, 1))
        dists = self.get_distances(x)
        arr = np.column_stack((self.instances, dists))
        arr = arr[arr[:,-1].argsort(kind='mergesort')] # Need a stable sorting algorithm for consistent results
        nearest = arr[:self.k, :-1]

        if both: return self.get_majority(nearest), self.prob_positive(nearest)
        if prob: self.prob_positive(nearest)
        return self.get_majority(nearest)
        
    def get_distances(self, x):
        X = np.array(list(x.T)*(self.m), dtype=np.float64)
        dists = np.sqrt(np.sum((self.instances[:, :-1] - X)**2, axis = 1)).reshape((self.m, 1))
        return dists
    
    def get_majority(self, arr):
        one_count = np.sum(arr[:, -1])
        zero_count = arr.shape[0] - one_count
        
        return 1 if one_count > zero_count else 0 # Predict 0 in case no majority

    def prob_positive(self, arr):
        return np.sum(arr[:, -1])/arr.shape[0]
import math

def euclidean_distance(x_1, x_2):
    if not isinstance(x_1, list) or not isinstance(x_2, list): return None
    if len(x_1) != len(x_2): return None

    return math.sqrt(sum([(x_1[i] - x_2[i])**2 for i in range(len(x_1))]))

class oneNN:
    def __init__(self, data):
        self.instances = data
    
    def evaluate(self, x):
        nearest = min(self.instances, key = lambda i: self.distance(i[:-1], x))
        return nearest[-1]

    def distance(self, p, x):
        return euclidean_distance(p, x)
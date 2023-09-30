import data_import
import oneNN
import matplotlib.pyplot as plt
from numpy import arange

if __name__ == '__main__':
    data = data_import.read_file_from_name('D2z.txt')
    classifier = oneNN.oneNN(data)

    grid_x = list(arange(-2, 2.1, 0.1))
    grid_y = list(arange(-2, 2.1, 0.1))
    grid = []
    for i in grid_x:
        for j in grid_y:
            grid.append([i, j])
            grid[-1].append(classifier.evaluate([i, j]))
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter([x1 for x1, x2, y in data if y == 1], [x2 for x1, x2, y in data if y == 1], color = 'black', marker = '+', s = 20, linewidths = 1, label='Training 1')
    plt.scatter([x1 for x1, x2, y in data if y == 0], [x2 for x1, x2, y in data if y == 0], color = 'black', marker = 'o', s = 20, linewidths = 1, label='Training 0', facecolors = 'none')
    plt.scatter([x1 for x1, x2, y in grid if y == 1], [x2 for x1, x2, y in grid if y == 1], color = 'red', s = 1, label='Testing 1')
    plt.scatter([x1 for x1, x2, y in grid if y == 0], [x2 for x1, x2, y in grid if y == 0], color = 'blue', s = 1, label='Testing 0')
    plt.legend()
    plt.show()
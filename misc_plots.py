import matplotlib.pyplot as plt

k = [1, 3, 5, 7, 10]
avg_acc = [0.8332, 0.8424, 0.8418, 0.8454, 0.8552]

plt.plot(k, avg_acc, marker = 'o', color='blue')
plt.grid(visible=True)
plt.xlabel('k')
plt.ylabel('Average accuracy')
plt.title('kNN 5-fold Cross validation')
plt.show()
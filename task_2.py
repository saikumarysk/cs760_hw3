import data_import
import oneNN
import kNN

# This takes a while to run. Roughly 70 mins to complete.
# The commented part where I use kNN (implemented using numpy) takes ~ 5 mins
# I learned my lesson. I will use numpy from here on.

# Results
# 0.825 0.65449 0.81754
# 0.853 0.68571 0.86643
# 0.862 0.72121 0.83803
# 0.851 0.71642 0.81633
# 0.775 0.60574 0.75817

if __name__ == '__main__':
    data = data_import.read_file_from_name('emails.csv', ',')

    for i in range(5):
        test_data = data[i*1000 : (i+1)*1000]
        training_data = data[0 : i*1000] + data[(i+1)*1000 : ]

        classifier = oneNN.oneNN(training_data)
        #classifier = kNN.kNN(1, training_data)
        tp, tn, fp, fn = 0, 0, 0, 0
        count = 0
        for instance in test_data:
            y_pred = classifier.evaluate(instance[:-1])
            y_actual = instance[-1]

            if y_pred == y_actual:
                if y_pred == 0: tn += 1
                else: tp += 1
            else:
                if y_actual == 1: fn += 1
                else: fp += 1
            
            count += 1
            print(count, '/', len(test_data), 'Done', end = '\r')
        
        accuracy = (tp + tn)/len(test_data)
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)

        print(round(accuracy, 5), round(precision, 5), round(recall, 5))
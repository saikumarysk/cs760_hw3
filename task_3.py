import data_import
import logistic

# Results
# 0.914 0.88417 0.80351
# 0.897 0.83206 0.787
# 0.883 0.89573 0.66549
# 0.82 0.92537 0.42177
# 0.851 0.79182 0.69608

if __name__ == '__main__':
    data = data_import.read_file_from_name('emails.csv', ',')

    for i in range(5):
        test_data = data[i*1000 : (i+1)*1000]
        training_data = data[0 : i*1000] + data[(i+1)*1000 : ]

        classifier = logistic.LogisticRegression(training_data)
        tp, tn, fp, fn = 0, 0, 0, 0
        count = 0
        for instance in test_data:
            y_pred = classifier.evaluate([1] + instance[:-1])
            
            y_pred = float(y_pred)
            y_actual = instance[-1]
            if y_pred == y_actual:
                if y_pred == 0: tn += 1
                else: tp += 1
            else:
                if y_actual == 1: fn += 1
                else: fp += 1
            
            count += 1
        
        accuracy = (tp + tn)/len(test_data)
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)

        print(round(accuracy, 5), round(precision, 5), round(recall, 5))
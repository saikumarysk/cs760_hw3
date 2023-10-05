import data_import
import kNN

# This also takes roughly 20 mins to complete.

# Results
# k = 1 : 0.825, 0.853, 0.862, 0.851, 0.775 - 0.8332
# k = 3 : 0.847, 0.851, 0.86, 0.88, 0.774 - 0.8424
# k = 5 : 0.838, 0.85, 0.873, 0.869, 0.779 - 0.8418
# k = 7 : 0.838, 0.862, 0.875, 0.874, 0.778 - 0.8454
# k = 10 : 0.862, 0.87, 0.876, 0.887, 0.781 - 0.8552

# Results after normalization - We might be skewing data and changing distances. This is wrong
# k = 1 : 0.832, 0.878, 0.883, 0.87, 0.821 - 0.8568
# k = 3 : 0.805, 0.845, 0.869, 0.836, 0.814 - 0.8338
# k = 5 : 0.766, 0.84, 0.846, 0.818, 0.799 - 0.8138
# k = 7 : 0.748, 0.801, 0.822, 0.812, 0.772 - 0.791
# k = 10 : 0.733, 0.777, 0.798, 0.797, 0.756 - 0.7722

if __name__ == '__main__':
    data = data_import.read_file_from_name('emails.csv', ',')

    for k in [1, 3, 5, 7, 10]:
        print('k is', k)
        accs = []
        for i in range(5):
            test_data = data[i*1000 : (i+1)*1000]
            training_data = data[0 : i*1000] + data[(i+1)*1000 : ]

            classifier = kNN.kNN(k, training_data)
            tp, tn, fp, fn = 0, 0, 0, 0
            count = 0
            for instance in test_data:
                y_pred = classifier.evaluate(instance[:-1])
                
                y_pred = float(y_pred)
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
            accs.append(accuracy)

            print("Accuracy is", round(accuracy, 5))
        
        print('Average Accuracy is', round(sum(accs)/5, 5))
import data_import
import ROC
import matplotlib.pyplot as plt
import logistic
import kNN

if __name__ == '__main__':
    data = data_import.read_file_from_name('emails.csv', delimiter=',')
    training_data = data[:4000]
    test_data = data[4000:]
    actual_y = [i[-1] for i in test_data]

    classifier_knn = kNN.kNN(5, training_data)
    classifier_logistic = logistic.LogisticRegression(training_data)

    print('Training Done!')

    knn_probs = []
    logistic_probs = []

    count = 0
    for instance in test_data:
        _, knn_prob = classifier_knn.evaluate(instance[:-1], both=True)
        knn_probs.append(knn_prob)
        count += 1
        print(f'{count} / 1000 Done!', end='\r')
    
    print('kNN Prediction Done!')
    
    count = 0
    for instance in test_data:
        _, logistic_prob = classifier_logistic.evaluate([1] + instance[:-1], both=True)
        logistic_probs.append(logistic_prob)
        count += 1
        print(f'{count} / 1000 Done!', end='\r')
    
    print('Logistic Regression Prediction Done!')

    roc_coords_knn, auc_knn = ROC.roc_auc(actual_y, knn_probs)
    roc_coords_logistic, auc_logistic = ROC.roc_auc(actual_y, logistic_probs)

    roc_x_knn = [fpr for fpr, tpr in roc_coords_knn]
    roc_y_knn = [tpr for fpr, tpr in roc_coords_knn] 

    roc_x_logistic = [fpr for fpr, tpr in roc_coords_logistic]
    roc_y_logistic = [tpr for fpr, tpr in roc_coords_logistic]
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid( visible = True )
    plt.plot(roc_x_knn, roc_y_knn, color = 'cyan', label = f'KNeighborsClassifier (AUC = {auc_knn})')
    plt.plot(roc_x_logistic, roc_y_logistic, color = 'orange', label = f'LogisticRegression (AUC = {auc_logistic})')
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")
    plt.legend()
    plt.show()

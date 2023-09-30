import matplotlib.pyplot as plt

def roc_auc(y_true, y_pred_probs):
    if len(y_true) != len(y_pred_probs): return None
    
    l = sorted([i for i in zip(y_pred_probs, y_true)], key = lambda i: -i[0])

    num_pos = sum(y_true)
    num_neg = len(y_true) - num_pos

    if num_pos == 0 : return [(0, 0), (1, 0)]
    elif num_neg == 0 : return [(0, 0), (0, 1)]

    TP, FP, last_TP = 0, 0, 0

    coords = [(0, 0)]
    for i in range(len(y_true)):

        if i > 1 and l[i][0] != l[i-1][0] and l[i][1] == 0 and TP > last_TP:
            coords.append((FP/num_neg, TP/num_pos))
            last_TP = TP
        
        if l[i][1] == 1: TP += 1
        else: FP += 1
    
    coords.append((FP/num_neg, TP/num_pos))

    return coords, auc(coords)

def auc(coords):

    result = 0
    for i in range(1, len(coords)):
        x1, y1 = coords[i-1]
        x2, y2 = coords[i]

        result += 0.5*( x2- x1 )*( y1 + y2 ) # Area of a trapezium
    
    return result

if __name__ == '__main__':
    roc_coords, auc_val = roc_auc([1, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1])

    roc_x = [fpr for fpr, tpr in roc_coords]
    roc_y = [tpr for fpr, tpr in roc_coords]

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid( visible = True )
    plt.plot(roc_x, roc_y, color = 'red', marker='o', label = f'ROC (AUC - {auc_val})')
    plt.title("ROC Curve for the Given Dataset")
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")
    plt.legend()
    plt.show()
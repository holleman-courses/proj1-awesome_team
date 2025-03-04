import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('hw4_data.csv')

output = data['model_output'].to_numpy()
prediction = data['prediction'].to_numpy()
true_class = data['true_class'].to_numpy()

tp =  np.sum(np.logical_and(prediction, true_class))
fp = np.sum( np.logical_and(prediction, np.logical_not(true_class)))
tn = np.sum( np.logical_and(np.logical_not(prediction), np.logical_not(true_class)))
fn =  np.sum(np.logical_and(np.logical_not(prediction), true_class))

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("True Positives: ", tp)
print("False Positives: ", fp)
print("True Negatives: ", tn)
print("False Negatives: ", fn)

print("Precision: ", precision)
print("Recall: ", recall)


def calc_pr(threshold: float, output: np.ndarray, targets: np.ndarray):
    
    pred = output > threshold
    
    tp =  np.sum(np.logical_and(pred, targets))
    fp = np.sum( np.logical_and(pred, np.logical_not(targets)))
    tn = np.sum( np.logical_and(np.logical_not(pred), np.logical_not(targets)))
    fn =  np.sum(np.logical_and(np.logical_not(pred), targets))

    # precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # f1 = (2 * precision * recall) / (precision + recall)

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    
    return (fpr, tpr, recall)

step = 1e-6
thresholds = np.arange(output.min(), output.max(), step)

tpr_curve = np.zeros(thresholds.shape)
fpr_curve = np.zeros(thresholds.shape)
recall_curve = np.zeros(thresholds.shape)

for i, thresh in enumerate(thresholds):
    fpr_curve[i], tpr_curve[i], recall_curve[i] = calc_pr(thresh, output, true_class)

#min fpr where recall > 0.9
min_fpr = fpr_curve[recall_curve >= 0.9].min()
print("Min FPR where recall >= 0.9: ", min_fpr)


plt.plot(thresholds, tpr_curve, label="True Positive Rate")
plt.plot(thresholds, fpr_curve, label="False Positive Rate")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig("plot.png")
plt.show()
    


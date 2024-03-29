import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from pyswarm import pso
from sklearn import metrics

# done 'MoteStrain', 'ItalyPowerDemand','Strawberry','Herring','GunPointAgeSpan',, 'ProximalPhalanxOutlineCorrect',
#             'ToeSegmentation1', 'ToeSegmentation2',

datasets = ['Herring']
for i in datasets:
    data_train = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TRAIN.tsv", sep='\t')
    data_test = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TEST.tsv", sep='\t')

    X_train = data_train.iloc[:, 1:].values
    y_train = data_train.iloc[:, 0].values
    X_test = data_test.iloc[:, 1:].values
    y_test = data_test.iloc[:, 0].values
    temp = len(np.unique(y_test))

    print(f'dataset "" {i} "" is running plz wait, number of class is {temp}.')
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.20, random_state=42)
    X_train, X_test1, y_train, y_test1 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    def objective_function(params):
        C, gamma = params
        svm_classifier = SVC(kernel='rbf', C=C, gamma=gamma)
        svm_classifier.fit(X_train, y_train)
        y_pred_svm1 = svm_classifier.predict(X_test1)
        return -roc_auc_score(y_test1, y_pred_svm1)

    lb = [1, 0.0001]
    ub = [55, 0.01]
    best_params, _ = pso(objective_function, lb, ub, swarmsize=20, maxiter=100)
    print(best_params)
    best_C, best_gamma = best_params
    optimized_svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True)
    optimized_svm.fit(X_train, y_train)
    y_pred_optimized = optimized_svm.predict(X_test)
    final_auc = roc_auc_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    f1 = f1_score(y_test, y_pred_optimized)
    for ii in range(len(y_test)):
        if y_test[ii]==1:
            y_test[ii] = 0
        if y_test[ii]==2:
            y_test[ii] = 1
    for ii in range(len(y_pred_optimized)):
        if y_pred_optimized[ii]==1:
            y_pred_optimized[ii] = 0
        if y_pred_optimized[ii]==2:
            y_pred_optimized[ii] = 1
    y_p = optimized_svm.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_p[:,1])

    # Create ROC curve plot
    plt.figure(f'{i}')
    plt.plot(fpr, tpr, color='b', label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'plot/{i}_roc_curve.jpg', format='jpg', dpi=300)

    results_df = pd.DataFrame({
        'Metric': ['auc', 'Precision', 'Recall', 'F1 Score','best_c','best_gamma'],
        'Value': [final_auc, precision, recall, f1, best_params[0], best_params[1]]
    })
    results_df.to_csv(f'result/{i}_svm_metrics_results.csv', index=False)

    print("Metrics saved to 'svm_metrics_results.csv'")
    print(f'Final AUC using optimized SVM: {final_auc:.4f} \n\n')


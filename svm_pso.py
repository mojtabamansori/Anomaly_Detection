import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from pyswarm import pso

# more 2 class ==> datasets = 'Beef', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
# done ==> datasets = 'Beef', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',

datasets = [
             'FordB',
            'Herring', 'InlineSkate',
            'ItalyPowerDemand', 'MoteStrain',
            'ProximalPhalanxOutlineCorrect', 'ToeSegmentation2']
for i in datasets:
    data_train = pd.read_csv(f"{i}/{i}_TRAIN.tsv", sep='\t')
    data_test = pd.read_csv(f"{i}/{i}_TEST.tsv", sep='\t')

    X_train = data_train.iloc[:, 1:].values
    y_train = data_train.iloc[:, 0].values
    X_test = data_test.iloc[:, 1:].values
    y_test = data_test.iloc[:, 0].values
    print(i,np.unique(y_test))
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.20, random_state=12)

    def objective_function(params):
        C, gamma = params
        svm_classifier = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
        svm_classifier.fit(X_train, y_train)
        y_pred_svm = svm_classifier.predict(X_test)
        return -roc_auc_score(y_test, y_pred_svm)

    lb = [1, 0.0001]
    ub = [55, 0.01]
    best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=100)
    print(best_params)
    best_C, best_gamma = best_params
    optimized_svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma, class_weight='balanced')
    optimized_svm.fit(X_train, y_train)
    y_pred_optimized = optimized_svm.predict(X_test)
    final_auc = roc_auc_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    f1 = f1_score(y_test, y_pred_optimized)

    results_df = pd.DataFrame({
        'Metric': ['auc', 'Precision', 'Recall', 'F1 Score'],
        'Value': [final_auc, precision, recall, f1]
    })
    results_df.to_csv(f'{i}_svm_metrics_results.csv', index=False)

    print("Metrics saved to 'svm_metrics_results.csv'")
    print(f'Final AUC using optimized SVM: {final_auc:.4f}')


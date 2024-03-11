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

    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score


    def objective_function(params):
        # Unpack parameters
        m = []
        for i3 in range(10):
            C, gamma, maxiteration, init_learning = params

            # Create an MLP classifier
            hidden_layer_sizes = (int(C),)  # You can adjust this based on your problem
            mlp_classifier = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=gamma,
                learning_rate='constant',
                learning_rate_init=init_learning,
                max_iter=int(maxiteration),
                random_state=None,
                tol=0.0001,
                verbose=False,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,  # Fraction of training data for validation
                beta_1=0.9,  # Adam optimizer parameter
                beta_2=0.999,  # Adam optimizer parameter
                epsilon=1e-05,  # Adam optimizer parameter
                max_fun=15000  # Maximum number of function evaluations
            )
            mlp_classifier.fit(X_train, y_train)
            y_pred_mlp = mlp_classifier.predict(X_test1)
            roc_auc_score(y_test1, y_pred_mlp)
            final_auc = roc_auc_score(y_test1, y_pred_mlp)
            m.append(final_auc)
        return -np.mean(np.array(m))


    lb = [1, 0.0001, 1, 0.0001]
    ub = [4, 0.01, 10, 0.01]
    best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=100)
    print(best_params)
    m = []
    for i3 in range(10):
        best_C, best_gamma, best_maxiteration, best_init_learning = best_params

        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=int(best_C),
            activation='relu',  # You can choose other activation functions
            solver='adam',  # Use 'adam' for large datasets, or 'lbfgs' for small datasets
            alpha=best_gamma,  # L2 regularization strength
            learning_rate='constant',  # Keep learning rate constant
            learning_rate_init=best_init_learning,  # Initial learning rate
            max_iter=int(best_maxiteration),  # Maximum number of iterations
            random_state=None,  # Set a random seed if needed
            tol=0.0001,  # Tolerance for convergence
            verbose=False,  # Set to True for debugging
            warm_start=False,  # Set to True to reuse previous solution
            momentum=0.9,  # Momentum for gradient updates
            nesterovs_momentum=True,  # Use Nesterov's momentum
            early_stopping=False,  # Enable early stopping
            validation_fraction=0.1,  # Fraction of training data for validation
            beta_1=0.9,  # Adam optimizer parameter
            beta_2=0.999,  # Adam optimizer parameter
            epsilon=1e-05,  # Adam optimizer parameter
            max_fun=15000  # Maximum number of function evaluations
        )

        mlp_classifier.fit(X_train, y_train)
        y_pred_optimized = mlp_classifier.predict(X_test)
        final_auc = roc_auc_score(y_test, y_pred_optimized)
        m.append(final_auc)
    print(m)
    print(np.mean(np.array(m)))
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
    # y_p = optimized_svm.predict_proba(X_test)
    # fpr, tpr, _ = metrics.roc_curve(y_test, y_p[:,1])
    #
    # # Create ROC curve plot
    # plt.figure(f'{i}')
    # plt.plot(fpr, tpr, color='b', label='ROC Curve')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    # plt.xlim([-0.005, 1.005])
    # plt.ylim([-0.005, 1.005])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.savefig(f'plot/{i}_roc_curve.jpg', format='jpg', dpi=300)
    #
    # results_df = pd.DataFrame({
    #     'Metric': ['auc', 'Precision', 'Recall', 'F1 Score','best_c','best_gamma'],
    #     'Value': [final_auc, precision, recall, f1, best_params[0], best_params[1]]
    # })
    # results_df.to_csv(f'result/{i}_svm_metrics_results.csv', index=False)
    #
    # print("Metrics saved to 'svm_metrics_results.csv'")
    # print(f'Final AUC using optimized SVM: {final_auc:.4f} \n\n')
    #

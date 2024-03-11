import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from pyswarm import pso
from torch.optim.lr_scheduler import ReduceLROnPlateau

datasets = ['Herring']
input_size = 512
num_epochs = 20  # Increased number of epochs for better convergence

# Define the hyperparameter search space
lower_bounds = [0.0001, 32, 1, 256, 150]  # Learning rate, batch size, num_epochs, hidden_size1, hidden_size2
upper_bounds = [0.001, 64, 10, 512, 256]

num_particles = 20
metric_values = []

# Split the data into training, validation, and test sets
for i in datasets:
    data_train = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TRAIN.tsv", sep='\t')
    data_test = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TEST.tsv", sep='\t')

    X_train, X_temp, y_train, y_temp = train_test_split(data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values, test_size=0.2, random_state=12)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=12)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the neural network model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.bn1 = nn.BatchNorm1d(hidden_size1)  # Added Batch Normalization
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.bn2 = nn.BatchNorm1d(hidden_size2)  # Added Batch Normalization
            self.fc3 = nn.Linear(hidden_size2, 1)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))  # Applied Batch Normalization and ReLU activation
            x = F.relu(self.bn2(self.fc2(x)))  # Applied Batch Normalization and ReLU activation
            x = self.fc3(x)
            return x

    def train_model(model, criterion, optimizer, scheduler, num_epochs, batch_size):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                inputs = torch.from_numpy(X_train[i:i + batch_size, :]).to(device).float()
                labels = torch.from_numpy(y_train[i:i + batch_size]).to(device).float()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / (len(X_train) / batch_size)
            scheduler.step(epoch_loss)

    def evaluate_model(model, X, y, batch_size):
        model.eval()
        y_scores = []
        y_true = []

        with torch.no_grad():  # Use no_grad() to disable gradient computation during evaluation
            for i in range(0, len(X), batch_size):
                inputs = torch.from_numpy(X[i:i + batch_size, :]).to(device).float()
                labels = torch.from_numpy(y[i:i + batch_size]).to(device).float()

                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs)

                y_scores.extend(probabilities.cpu().detach().numpy())
                y_true.extend(labels.cpu().detach().numpy())

        return y_true, y_scores

    def objective(params):
        learning_rate, batch_size, num_epochs, hidden_size1, hidden_size2 = params

        model = NeuralNet(input_size, int(hidden_size1), int(hidden_size2)).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min')  # Learning rate scheduler
        train_model(model, criterion, optimizer, scheduler, int(num_epochs), int(batch_size))
        y_true_val, y_scores_val = evaluate_model(model, X_val, y_val, int(batch_size))

        auc = roc_auc_score(y_true_val, y_scores_val)
        precision = precision_score(y_true_val, (np.array(y_scores_val) > 0.5).astype(int), average='weighted',
                                    zero_division=1)
        recall = recall_score(y_true_val, (np.array(y_scores_val) > 0.5).astype(int), average='weighted',
                              zero_division=1)

        f1 = f1_score(y_true_val, (np.array(y_scores_val) > 0.5).astype(int), average='weighted',
                              zero_division=1)
        accuracy = accuracy_score(y_true_val, (np.array(y_scores_val) > 0.5).astype(int))

        # You can adjust weights based on your priorities
        metric_combination = 0.2 * auc + 0.2 * precision + 0.2 * recall + 0.2 * f1 + 0.2 * accuracy
        metric_values.append(metric_combination)

        print(f'Metric Combination: {metric_combination}, AUC: {auc}, Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}')
        return -metric_combination

    start_time = time.time()
    best_params, _ = pso(objective, lower_bounds, upper_bounds, swarmsize=num_particles, maxiter=100)
    elapsed_time = time.time() - start_time

    print(f"Elapsed Time: {elapsed_time} seconds")

    model = NeuralNet(input_size, int(best_params[3]), int(best_params[4])).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params[0])
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for _ in range(10):
        train_model(model, criterion, optimizer, scheduler, num_epochs, int(best_params[1]))
        y_true_test, y_scores_test = evaluate_model(model, X_test, y_test, int(best_params[1]))

        threshold = 0.5  # You can adjust this threshold based on your needs
        y_pred_binary = (np.array(y_scores_test) > threshold).astype(int)
        auc = roc_auc_score(y_true_test, y_scores_test)
        precision = precision_score(y_true_test, (np.array(y_scores_test) > 0.5).astype(int), average='weighted',
                                    zero_division=1)
        recall = recall_score(y_true_test, (np.array(y_scores_test) > 0.5).astype(int), average='weighted',
                              zero_division=1)

        f1 = f1_score(y_true_test, (np.array(y_scores_test) > 0.5).astype(int), average='weighted',
                      zero_division=1)
        auc_test = roc_auc_score(y_true_test, y_scores_test)

        accuracy = accuracy_score(y_test, y_pred_binary)
        print(f'Final AUC on the test set: {auc_test}, Accuracy: {accuracy},{f1},{recall},{precision}')
        print(f'Final AUC on the test set: {auc_test}')

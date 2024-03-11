import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch
import torch.nn as nn
from pyswarm import pso

datasets = ['Herring']
input_size = 512
num_epochs = 1

# Define the hyperparameter search space
lower_bounds = [0.0001, 2, 1, 511, 255]  # Learning rate, batch size, num_epochs, hidden_size1, hidden_size2
upper_bounds = [0.001, 12, 2, 512, 256]

# Number of particles in the population
num_particles = 10

# Number of dimensions (hyperparameters)
num_dimensions = len(lower_bounds)

# Store AUC values for each run
auc_values = []

# Perform 10 runs

# Split the data into training, validation, and test sets
for i in datasets:
    data_train = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TRAIN.tsv", sep='\t')
    data_test = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TEST.tsv", sep='\t')

    X_train, X_temp, y_train, y_temp = train_test_split(data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the neural network model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.sigmoid(out)
            return out

    def objective(params):
        # Unpack the hyperparameters
        learning_rate, batch_size, num_epochs, hidden_size1, hidden_size2 = params

        # Modify y_train to be in the range [0, 1]
        y_train_modified = (y_train > 1).astype(float)

        model = NeuralNet(input_size, int(hidden_size1), int(hidden_size2)).to(device)

        # Define loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(int(num_epochs)):
            total_loss = 0.0
            for i2 in range(0, len(X_train) - 1, int(batch_size)):
                x = torch.from_numpy(X_train[i2:i2 + int(batch_size), :]).to(device).float()
                y = torch.from_numpy(y_train_modified[i2:i2 + int(batch_size)]).to(device).float()

                outputs = model(x)
                loss = criterion(outputs, y.view(-1, 1))
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_loss = total_loss / (len(X_train) / int(batch_size))

        with torch.no_grad():
            y_scores = []
            y_true = []

            for i2 in range(0, len(X_val) - 1, int(batch_size)):
                x = torch.from_numpy(X_val[i2:i2 + int(batch_size), :]).to(device).float()
                y = torch.from_numpy(y_val[i2:i2 + int(batch_size)]).to(device).float()

                outputs = model(x)
                probabilities = torch.sigmoid(outputs)

                y_scores.extend(probabilities.cpu().numpy())
                y_true.extend(y.cpu().numpy())

            auc = roc_auc_score(y_true, y_scores)
            print(f'Run {1}: Learning Rate: {learning_rate}, Batch Size: {batch_size}, Num Epochs: {int(num_epochs)}, Hidden Size 1: {int(hidden_size1)}, Hidden Size 2: {int(hidden_size2)}')
            print(f'AUC on the validation set: {auc}')
            auc_values.append(auc)

        return -auc  # We're using PSO to maximize AUC, so we minimize its negative value

    # PSO optimization
    start_time = time.time()
    best_params, _ = pso(objective, lower_bounds, upper_bounds, swarmsize=num_particles, maxiter=50)
    elapsed_time = time.time() - start_time



    # Print elapsed time
    print(f"Elapsed Time: {elapsed_time} seconds")

    # After finding the best_params, use them to train the model on the full training set
    # Evaluate the model on the test set and report the final AUC
    with torch.no_grad():
        y_scores = []
        y_true = []
        for run in range(10):
            for i2 in range(0, len(X_test) - 1, int(best_params[1])):
                x = torch.from_numpy(X_test[i2:i2 + int(best_params[1]), :]).to(device).float()
                y = torch.from_numpy(y_test[i2:i2 + int(best_params[1])]).to(device).float()
                model = NeuralNet(input_size, int(best_params[3]), int(best_params[4])).to(device)
                outputs = model(x)
                probabilities = torch.sigmoid(outputs)
                y_scores.extend(probabilities.cpu().numpy())
                y_true.extend(y.cpu().numpy())

            auc = roc_auc_score(y_true, y_scores)
            print(f'Final AUC on the test set: {auc}')
            auc_values.append(auc)

# Calculate and print the average AUC over 10 runs
average_auc = np.mean(auc_values)
print(f'Average AUC over 10 runs: {average_auc}')

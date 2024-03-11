import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
import torch.nn as nn

datasets = ['Herring']
input_size = 512
hidden_size = 500
num_epochs = 100
batch_size = 5
learning_rate = 0.001
for num_epochs in range(1,50,10):
    for hidden_size in range(480,500,9):
        for i in datasets:
            data_train = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TRAIN.tsv", sep='\t')
            data_test = pd.read_csv(f"E:/python/UCRArchive_2018/{i}/{i}_TEST.tsv", sep='\t')

            X_train = data_train.iloc[:, 1:].values
            y_train = data_train.iloc[:, 0].values
            X_test = data_test.iloc[:, 1:].values
            y_test = data_test.iloc[:, 0].values
            temp = len(np.unique(y_test))

            X_combined = np.vstack((X_train, X_test))
            y_combined = np.concatenate((y_train, y_test))
            num_classes = len(np.unique(y_combined))  # Update the number of classes dynamically

            X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.20, random_state=42)
            X_train, X_test1, y_train, y_test1 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

            # Check Device configuration
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Fully connected neural network
            class NeuralNet(nn.Module):
                def __init__(self, input_size, hidden_size, num_classes):
                    super(NeuralNet, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_size, num_classes)

                def forward(self, x):
                    out = self.fc1(x)
                    out = self.relu(out)
                    out = self.fc2(out)
                    return out  # Return raw logits without softmax

            model = NeuralNet(input_size, hidden_size, num_classes).to(device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Convert y_train to integer type
            y_train = y_train.astype(int)
            for iy in range(len(y_train)):
                if y_train[iy]==1:
                    y_train[iy] =0
                if y_train[iy] == 2:
                    y_train[iy] = 1
            for epoch in range(num_epochs):
                for i2 in range(0, len(X_train) - 1, batch_size):
                    x = torch.from_numpy(X_train[i2:i2 + 5, :]).to(device).float()
                    y = torch.from_numpy(y_train[i2:i2 + 5]).to(device).long()  # Convert to torch.LongTensor
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                correct = 0
                total = 0
                for i2 in range(0, len(X_test) - 1, batch_size):
                    x = torch.from_numpy(X_test[i2:i2 + 5, :]).to(device).float()
                    y = torch.from_numpy(y_test[i2:i2 + 5]).to(device).long()  # Convert to torch.LongTensor
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                print(f'epoch={num_epochs}-hidensize={hidden_size}')
                print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))

            # Save the model checkpoint
            torch.save(model.state_dict(), 'model.ckpt')

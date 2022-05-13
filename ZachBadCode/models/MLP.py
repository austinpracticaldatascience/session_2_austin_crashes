
"""
Find source at: 
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_fullynet.py
"""

# Imports
import torch
import torchvision 
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn 
from torch.utils.data import DataLoader  
from tqdm import tqdm 
import numpy as np

#Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparameters of our neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 542
num_classes = 2
learning_rate = 3e-4
batch_size = 128
num_epochs = 1


def train(data, labels):
    # Load Training and Test data
    dataset = np.asarray(zip(data, labels))
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = NN(input_size=input_size, num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Get to correct shape
            data = data.reshape(data.shape[0], -1)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples




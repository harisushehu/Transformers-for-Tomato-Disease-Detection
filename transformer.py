#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:38:52 2024

@author: harisushehu
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the dataset
data_dir = "../PlantVillage"
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

# Calculate the sizes of each dataset
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Apply the appropriate transformations
val_dataset.dataset.transform = data_transforms['val_test']
test_dataset.dataset.transform = data_transforms['val_test']

# Create DataLoader instances
batch_size = 32
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
}
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}
class_names = full_dataset.classes


# Define the Transformer-based model
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=10, num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.token_embedding = nn.Linear(3 * 224 * 224, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 256, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.token_embedding(x).unsqueeze(1) + self.positional_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = self.layer_norm(x.mean(dim=1))
        x = self.fc(x)
        return x


# Training function
def train_model(model, train_loader, val_loader, save_path='best_model.pth', num_epochs=100, lr=1e-4, verbose=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes['train']
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}')

        # Save the model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved model with validation accuracy: {val_acc:.4f}')

    return best_acc



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    best_acc = 0.0
    best_model_path = 'best_model.pth'
    num_epochs = 10

    # Initialize lists to track results
    results = []

    for run in range(1):  #10
        model = TransformerClassifier(num_classes=len(class_names)).to(device)
        save_path = f'model_run_{run + 1}.pth'  # Unique file name for each run
        print(f'Run {run + 1}/10')

        val_acc = train_model(model, dataloaders['train'], dataloaders['val'], num_epochs=num_epochs, lr=1e-4, verbose=1)
        print(f'Run {run + 1}/10, Final Validation Accuracy: {val_acc:.4f}')

        # Update the best model if the current run has a better validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = save_path
            print(f'New best model with validation accuracy: {val_acc:.4f}')

        # Evaluate the current model on the test set
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        print(f'Run {run + 1}/10, Test Accuracy: {test_acc:.4f}')

        # Append the results for this run
        results.append({'Run': run + 1, 'Validation Accuracy': val_acc, 'Test Accuracy': test_acc})

    # Save the best model separately
    #best_model = TransformerClassifier(num_classes=len(class_names)).to(device)
    #best_model.load_state_dict(torch.load(best_model_path))
    #torch.save(best_model.state_dict(), 'best_model_final.pth')

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("PlantVillage_results.csv", index=False)
















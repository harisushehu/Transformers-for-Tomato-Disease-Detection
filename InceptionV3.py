# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 2024

@author: harisushehu
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import os

'''
CUDA_VISIBLE_DEVICES=0 python InceptionV3.py
'''

# Directory paths
base_dir = "../../data/DIKUMARI"

# Data transforms
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 expects 299x299 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=base_dir, transform=transform)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.inception_v3(pretrained=False, num_classes=len(dataset.classes), aux_logits=True).to(device)
model = torch.nn.DataParallel(model)  # Optional: if you have multiple GPUs
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Handle the case where InceptionV3 returns a tuple
            if isinstance(outputs, tuple):
                outputs, _ = outputs  # Unpack only if necessary
            # Calculate loss using the main output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs  # Unpack only if necessary
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_acc = val_corrects / val_total
        print(f'Validation Accuracy: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'inceptionv3_best.pth')

    print(f'Best Validation Accuracy: {best_acc:.4f}')
    return best_acc

# Training
start_time = time.time()
best_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer)
end_time = time.time()

# Evaluation
model.load_state_dict(torch.load('inceptionv3_best.pth'))
model.eval()
test_corrects = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs, _ = outputs  # Unpack only if necessary
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data).item()
        test_total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_corrects / test_total
conf_matrix = confusion_matrix(all_labels, all_preds)
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
time_str = f"{int(hours)}:{int(minutes)}:{int(seconds)}"

# Save results to CSV
results_df = pd.DataFrame({
    'Iteration': [1],
    'Accuracy': [test_acc],
    'Time': [time_str]
})
results_df.to_csv('InceptionV3_result.csv', index=False)

print(f"Accuracy: {test_acc}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Time Elapsed: {time_str}")

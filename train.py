#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:32:21 2024

@author: harisushehu
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Function to train the model
def train_model(model, train_loader, val_loader, dataset_sizes, device, save_path='best_model.pth', num_epochs=100, lr=1e-4, verbose=1):
    """
    Trains the model and evaluates its performance.
    """
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
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes['train']
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_preds = []  # Initialize as empty list
        val_labels = []  # Initialize as empty list
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                _, preds = torch.max(logits, 1)
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

# Function to load the Vision Transformer (ViT) model
def load_vit_model(weights_choice, class_names):
    """
    Loads a Vision Transformer model with the specified weights.
    """
    from PIL import Image
    import requests
    from timm import create_model
    from transformers import ViTImageProcessor, ViTForImageClassification

    if weights_choice == 'vit_base_patch16_224':
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    elif weights_choice == 'vit_small_patch16_224':
        img = Image.open(requests.get(
            'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png', stream=True).raw)
        model = create_model('vit_small_patch16_224.augreg_in21k', pretrained=True)
    else:
        model = create_model('vit_base_patch16_224', pretrained=True, num_classes=len(class_names))
    return model


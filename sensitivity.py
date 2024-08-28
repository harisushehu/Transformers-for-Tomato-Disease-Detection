#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:03:13 2024

@author: harisushehu
"""


import torch
import pandas as pd
import random
from dataloader import load_data
from train import train_model, load_vit_model
from sklearn.metrics import accuracy_score

# Define parameter choices for sensitivity analysis
batch_sizes = [16, 32, 64]
epochs_list = [3, 5, 10]
learning_rates = [1e-4, 1e-3, 1e-2]

# Device configuration
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Load data
data_dir = "../../PlantVillage"
dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size=32)  # Initial batch size for data loading

all_results = []

# Sensitivity analysis across different parameter choices
for batch_size in batch_sizes:
    for num_epochs in epochs_list:
        for lr in learning_rates:
            for run in range(1):  # Number of runs can be adjusted as needed
                print(f'Run {run + 1} with Batch Size: {batch_size}, Epochs: {num_epochs}, Learning Rate: {lr}')

                # Reload data with the current batch size
                dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size=batch_size)

                model = load_vit_model("vit_base_patch16_224", class_names).to(device)
                save_path = f"./results/model_run_{run + 1}_bs{batch_size}_epochs{num_epochs}_lr{lr}.pth"

                val_acc = train_model(model, dataloaders['train'], dataloaders['val'], dataset_sizes, device, save_path=save_path, num_epochs=num_epochs, lr=lr, verbose=1)
                print(f'Final Validation Accuracy: {val_acc:.4f}')

                # Evaluate the model on the test set
                model.eval()
                test_preds, test_labels = [], []
                with torch.no_grad():
                    for inputs, labels in dataloaders['test']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)

                        # Check if outputs is an instance of a specific class and extract logits
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        else:
                            logits = outputs

                        _, preds = torch.max(logits, 1)
                        test_preds.extend(preds.cpu().numpy())
                        test_labels.extend(labels.cpu().numpy())


                test_acc = accuracy_score(test_labels, test_preds)
                print(f'Test Accuracy: {test_acc:.4f}')

                # Save the results for this run
                results = {
                    'Run': run + 1,
                    'Batch Size': batch_size,
                    'Epochs': num_epochs,
                    'Learning Rate': lr,
                    'Validation Accuracy': val_acc,
                    'Test Accuracy': test_acc
                }
                all_results.append(results)

# Save all results to a single CSV file
results_df = pd.DataFrame(all_results)
results_csv_path = "./results/sensitivity_analysis_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f'Saved results to {results_csv_path}')

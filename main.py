#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:03:13 2024

@author: harisushehu
"""

import torch
import pandas as pd
import argparse
import random
from dataloader import load_data
from train import train_model, load_vit_model
from sklearn.metrics import accuracy_score

'''

# To run using command, e.g. with all commands

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data-dir "../../PlantVillage" \
    --batch-size 32 \
    --num-epochs 10 \
    --model "vit_base_patch16_224" \
    --lr 1e-4 \
    --out-dir "./results" \
    --seed 42 \
    --num-runs 10

e.g. with few commands
CUDA_VISIBLE_DEVICES=0 python main.py --model "vit_base_patch16_224" --num-epochs 10 --batch-size 32 --lr 1e-4 --out-dir "./results" --seed 42

'''

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model on PlantVillage dataset.")
    parser.add_argument('--data-dir', type=str, default="../../data/DIKUMARI", help='Directory for the dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--model', type=str, choices=['vit_base_patch16_224', 'vit_small_patch16_224', 'imagenet'], default='vit_base_patch16_224', help='Model to use for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--out-dir', type=str, default='./', help='Output directory to save the model and results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs to perform for each model')
    return parser.parse_args()

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataloaders, dataset_sizes, class_names = load_data(args.data_dir, args.batch_size)

    best_acc = 0.0
    best_model_choice = args.model
    all_results = []

    # Run training multiple times
    for run in range(args.num_runs):
        print(f'Run {run + 1}/{args.num_runs} with weights: {args.model}')
        model = load_vit_model(args.model, class_names).to(device)
        save_path = f"{args.out_dir}/model_{args.model}_run{run + 1}.pth"

        val_acc = train_model(model, dataloaders['train'], dataloaders['val'], dataset_sizes, device, save_path=save_path, num_epochs=args.num_epochs, lr=args.lr, verbose=1)
        print(f'Final Validation Accuracy with {args.model} weights: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_choice = args.model
            best_model_path = save_path
            print(f'New best model with validation accuracy: {val_acc:.4f}')

        # Evaluate the model on the test set
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                _, preds = torch.max(logits, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        print(f'Test Accuracy with {args.model} weights: {test_acc:.4f}')

        # Save the results for this run
        results = {'Run': run + 1, 'Weights': args.model, 'Validation Accuracy': val_acc, 'Test Accuracy': test_acc}
        all_results.append(results)

    # Save all results to a single CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_path = f"{args.out_dir}/{args.model}_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f'Saved results to {results_csv_path}')

    # Save the best model separately
    best_model = load_vit_model(best_model_choice, class_names).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    torch.save(best_model.state_dict(), f"{args.out_dir}/{args.model}_best_model_final.pth")
    print(f'Saved best model to {args.out_dir}/{args.model}_best_model_final.pth')

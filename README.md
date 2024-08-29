# Transfer Learning with Transformers for Early Detection of Tomato Leaf Diseases

## Overview

This repository contains the codebase used for our paper titled **"Transfer Learning with Transformers for Early Detection of Tomato Leaf Diseases"**. This paper explores the application of transfer learning techniques using transformers and other state-of-the-art models for early detection of tomato leaf diseases. If you use our code or dataset in your research, please cite our paper (see BibTeX citation below).

## Code Structure

### Baseline Models

1. **Baseline Model without Attention Mechanism**  
   To run the baseline model without the attention mechanism, execute:

```bash
   CUDA_VISIBLE_DEVICES=0 python baseline.py
  ```

2. **Baseline Model with Attention Mechanism**  
   To run the baseline model with the attention mechanism, execute:

```bash
   CUDA_VISIBLE_DEVICES=0 python transformer.py
  ```

### State-of-the-Art Models

To obtain results using state-of-the-art models, you can run the respective scripts (`VGG19.py`, `ResNet50.py`, `InceptionV3.py`, or `EfficientNetB2.py`) for each model. Below is an example command for running the **EfficientNet** model. You can use similar commands for other models by replacing the script name.

- **EfficientNet**  
  To run the EfficientNet model, execute:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python EfficientNetB2.py
  ```

### Proposed Method

1. **Sensitivity Analysis**
To perform a sensitivity analysis and find the best hyperparameters that yield the highest accuracy, execute:

```bash
CUDA_VISIBLE_DEVICES=0 python sensitivity.py
```

## Data Loading and Preprocessing

### Data Loader
The `dataloader.py` file handles data preprocessing, including data augmentation, and loads the dataset for training and testing.

### Transfer Learning with Pretrained Models

#### Transfer Learning Setup
The `train.py` script loads pretrained models on ImageNet to apply transfer learning. We use three different sets of weights:
1. **ImageNet** with limited parameter tuning.
2. **Google's vit_base_patch16_224**, pretrained on ImageNet-21k and fine-tuned on ImageNet2012.
3. **vit_small_patch16_224**, pretrained with ImageNet-21k with additional augmentation and regularization.

#### Main Training Script
The `main.py` script compiles and runs the code to train, validate, and test on the specified dataset. To run this script, you may use specific commands:

## Example of a Basic Command

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model "vit_base_patch16_224" --num-epochs 10 --batch-size 32 --lr 1e-4 --out-dir "./results" --seed 42
```
This command specifies the GPU slot, model, number of epochs, batch size, learning rate, output directory, and seed number.

## Example with all relevant parameters:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data-dir "../../PlantVillage" \
    --batch-size 32 \
    --num-epochs 10 \
    --model "vit_base_patch16_224" \
    --lr 1e-4 \
    --out-dir "./results" \
    --seed 42 \
    --num-runs 10
```

### Citation

If you use this code or dataset in your research, please cite our paper:

```bibitex
@article{Shehu2024TomatoLeafDiseases,
  title={Transfer Learning with Transformers for Early Detection of Tomato Leaf Diseases},
  author={Shehu, Harisu Abdullahi and Ackley, Aniebietabasi and Mark, Marvellous and Eteng, Ofem},
  journal={Under Review},
  year={2024}
}
```

For any questions or inquiries, please contact:
Harisu Shehu (harisushehu@ecs.vuw.ac.nz)











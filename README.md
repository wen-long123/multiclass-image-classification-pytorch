# Image Classification with Custom MLP, Custom and Pretrained VGG16 Models

## Project Overview

This project implements and compares three different deep learning models for multi-class image classification:

1. **Custom Multilayer Perceptron (MLP) Model**
2. **Custom VGG16 Convolutional Neural Network (CNN) Model (from scratch)**
3. **Pretrained VGG16 Model (using ImageNet weights)**

The goal is to evaluate the effectiveness of deep CNNs and transfer learning on a dataset of 8 classes and compare them with a simpler MLP baseline.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Models Implemented](#models-implemented)  
- [Dataset](#dataset)  
- [Training Setup](#training-setup)  
- [Hyperparameter Tuning](#hyperparameter-tuning)  
- [Evaluation](#evaluation)  
- [Test on Unknown Data](#test-on-unknown-data)  
- [Results and Analysis](#results-and-analysis)  
- [Visualization](#visualization)  
- [How to Run](#how-to-run)  

---

## Project Structure

- `model_mlp.py` — Custom MLP implementation  
- `model_vgg16_custom.py` — Custom VGG16 model built from scratch with batch normalization  
- `model_vgg16_pretrained.py` — VGG16 model initialized with pretrained ImageNet weights, fine-tuned on custom dataset  
- `train.py` — Training scripts including hyperparameter tuning  
- `evaluate.py` — Metrics calculations and confusion matrix plotting  
- `test_unknown.py` — Testing models on unseen images  
- `data_processing.py` — Dataset preprocessing and transformations  
- `utils.py` — Helper functions (e.g., visualization, tensorboard logging)  

---

## Models Implemented

### 1. Custom MLP Model

- Simple fully connected layers without convolutional layers.
- Suitable for baseline performance on structured data.
- Limited capability in learning spatial features of images.

### 2. Custom VGG16 Model

- Replicates VGG16 architecture from scratch using PyTorch.
- 13 convolutional layers grouped into 5 blocks with max pooling.
- Batch normalization added after each convolutional layer.
- 3 fully connected layers with dropout for classification.
- Trained with SGD optimizer with momentum.

### 3. Pretrained VGG16 Model

- Utilizes PyTorch's torchvision `vgg16(pretrained=True)` model.
- Final classifier layer replaced to match number of classes (8).
- Fine-tuned with Adam optimizer.
- Leverages powerful feature extraction from ImageNet pretrained weights.

---

## Dataset

- 8-class image dataset (categories defined in `categories` variable).
- Images resized and normalized to 3×224×224 RGB format.
- Dataset split into training and validation sets.
- Data augmentations and preprocessing applied consistently.

---

## Training Setup

- Input image size: 224×224 pixels, RGB.
- Batch size, epochs, and device configuration set in training scripts.
- Training runs performed for 40 epochs per model.
- Optimizers and hyperparameters chosen as per model requirements.

---

## Hyperparameter Tuning

- Grid search over combinations of learning rate and weight decay for each model:
  - Learning rates: `[0.01, 0.001, 0.0001]`
  - Weight decay: `[0.0001, 0.0005]`
- Momentum fixed at `0.9` for SGD optimizer.
- Best parameters selected based on validation accuracy, loss, and Mean Average Precision (MAP).

---

## Evaluation

- Metrics calculated include:
  - Validation accuracy
  - Validation loss
  - Mean Average Precision (MAP)
  - Confusion matrix visualization
- Predictions on random validation batches compared to ground truth.

---

## Test on Unknown Data

- Tested all models on a fixed unknown test image (`test7.jpg`) from the `Test_path/unknown` directory.
- Same preprocessing pipeline applied.
- Predictions compared side by side to assess generalization.

---

## Results and Analysis

| Model                  | Learning Rate | Weight Decay | Mean Average Precision (MAP) | Test Sample Prediction |
|------------------------|---------------|--------------|------------------------------|------------------------|
| **Pretrained VGG16**   | 0.0001        | 0.0001       | **0.9870**                   | `knitwear`             |
| **Custom VGG16**       | 0.001         | 0.0005       | 0.9677                       | `tees`                 |
| **Custom MLP**         | 0.0001        | 0.0001       | 0.8889                       | `knitwear`             |

**Insights:**

- The pretrained VGG16 model outperformed others due to transfer learning benefits from large-scale ImageNet features.
- Custom VGG16 trained from scratch performed well but required more data and time.
- MLP model struggled to capture spatial features, resulting in lower performance.

---

## Visualization

- Confusion matrices plotted for each model.
- Predictions vs ground truth images shown for validation examples.
- TensorBoard used for monitoring training loss, accuracy, and other metrics.

---

## How to Run

steps:
  - "Download the project files to your local machine."
  - "Upload the files into your Google Colab environment."
  - "Run the notebook cells one by one in Colab to train, evaluate, and test the models."
note: "No additional setup is required — all dependencies and configurations are handled within the notebook."

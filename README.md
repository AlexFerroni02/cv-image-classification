# Image Classification with Deep Learning (CIFAR-10)

## Overview
This project focuses on building, training, and analyzing a deep learning model
for image classification using the CIFAR-10 dataset.

The main goal is not only to achieve good accuracy, but to deeply understand
model behavior through systematic error analysis and interpretability techniques.

## Objectives
- Build a CNN from scratch using PyTorch
- Train the model in a reproducible way
- Improve performance through regularization and data augmentation
- Perform detailed error analysis
- Interpret model predictions using Grad-CAM

## Dataset
CIFAR-10 consists of 60,000 32x32 color images across 10 classes.
It is a standard benchmark for image classification tasks.

## Project Structure
cv-image-classification/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_error_analysis.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   └── metrics.json
│
├── requirements.txt
├── README.md
└── .gitignore

## Model
The baseline model is a custom Convolutional Neural Network composed of:
- 3 convolutional blocks (Conv + ReLU + Pool)
- Fully connected classifier
- Trained using CrossEntropy loss and Adam optimizer

## Experiments
Several experiments were conducted:
- Baseline training
- Data augmentation
- Regularization (Dropout, BatchNorm)
- Learning rate scheduling

## Evaluation & Error Analysis
Beyond accuracy, the following analyses were performed:
- Confusion matrix
- Class-wise accuracy
- Visualization of misclassified samples
- Grad-CAM for model interpretability

## Results
Final results and visualizations are available in the `results/` folder.

## Technologies
- Python
- PyTorch
- torchvision
- OpenCV
- Google Colab

## Author
Alex

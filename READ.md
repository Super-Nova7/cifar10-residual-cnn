CIFAR-10 Residual CNN
Project Overview
This project implements a custom Convolutional Neural Network (CNN) with Residual (Skip) Connections for image classification on the CIFAR-10 dataset using TensorFlow and Keras.
The goal of this project was to understand how residual connections work and how they help deeper neural networks train more effectively.
Dataset
The model is trained on the CIFAR-10 dataset.
The dataset contains 60,000 color images of size 32x32 belonging to 10 classes:
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
Training Images: 48,000
Test Images: 12,000
Model Architecture
The network consists of:
Convolutional Layers
Batch Normalization
ReLU Activation
Residual (Skip) Connections
Max Pooling
Global Average Pooling
Fully Connected Layers
Residual blocks were implemented manually instead of using a prebuilt ResNet model.
Training Results
Training Accuracy: ~78%
Validation Accuracy: ~69%
Dataset: CIFAR-10
Epochs: 5
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Technologies Used
Python
TensorFlow
Keras
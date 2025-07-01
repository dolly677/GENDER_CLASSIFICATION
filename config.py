import torch
import os

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = r'C:\Users\dolly\finalproject\Gender_classification\Comsys_Hackathon5\Task_A'
model_save_path = 'best_model/gender_classification_transfer_learning_with_ResNet18.pth'

# Training parameters
batch_size = 16
num_epochs = 10
learning_rate = 0.001
momentum = 0.9

# Image transformations
image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

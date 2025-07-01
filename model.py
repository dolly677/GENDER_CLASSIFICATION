import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from config import device

def initialize_model():
    """Initialize and return the model"""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # binary classification
    model = model.to(device)
    return model

def load_saved_model(path):
    """Load a saved model from path"""
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    return model
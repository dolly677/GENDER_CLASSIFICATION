import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
from model import initialize_model
from data_loader import get_dataloaders
from utils import imshow
from config import *


def train_model():
    # Initialize model and dataloaders
    model = initialize_model()
    train_loader, val_loader, class_names = get_dataloaders()

    # Print model summary
    summary(model, input_size=(3, image_size, image_size))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        # Training phas
        model.train()
        train_loss, train_correct = 0., 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels.data)

        # Validation phase
        val_loss, val_correct = validate_model(model, val_loader, criterion)

        # Print epoch statistics
        print_epoch_stats(epoch, train_loss, train_correct, val_loss, val_correct,
                          len(train_loader.dataset), len(val_loader.dataset), start_time)

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Model saved successfully to: {model_save_path}")


def validate_model(model, val_loader, criterion):
    """Validation phase"""
    model.eval()
    val_loss, val_correct = 0., 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_correct += torch.sum(preds == labels.data)

    return val_loss, val_correct


def print_epoch_stats(epoch, train_loss, train_correct, val_loss, val_correct,
                      train_size, val_size, start_time):
    """Print training and validation statistics"""
    train_loss = train_loss / train_size
    train_acc = train_correct / train_size * 100.
    val_loss = val_loss / val_size
    val_acc = val_correct / val_size * 100.

    print(f'[Epoch {epoch + 1}/{num_epochs}]')
    print(f'Train - Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
    print(f'Valid - Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
    print(f'Time: {time.time() - start_time:.2f}s\n')


if __name__ == '__main__':
    train_model()
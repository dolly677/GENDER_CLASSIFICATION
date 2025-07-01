import device
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


def imshow(input, title):
    """Display image for Tensor"""
    input = input.numpy().transpose((1, 2, 0))
    input = np.clip(input * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    plt.imshow(input)
    plt.title(title)
    plt.show()


def visualize_predictions(model, dataloader, class_names, num_images=8):
    """Visualize model predictions"""
    model.eval()
    inputs, labels = next(iter(dataloader))
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    print('[Prediction Result Examples]')
    images = torchvision.utils.make_grid(inputs[:num_images // 2].cpu())
    imshow(images, title=[f'True: {class_names[x]}\nPred: {class_names[y]}'
                          for x, y in zip(labels[:num_images // 2], preds[:num_images // 2])])

    images = torchvision.utils.make_grid(inputs[num_images // 2:num_images].cpu())
    imshow(images, title=[f'True: {class_names[x]}\nPred: {class_names[y]}'
                          for x, y in zip(labels[num_images // 2:num_images], preds[num_images // 2:num_images])])
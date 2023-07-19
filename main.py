import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import random
import pandas as pd
from dataset import CustomDataset
from utils import preprocess, train, validate, test
from model import PretrainedResNet50

# set the seed for reproducibility
random_state = 42
np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)

batch_size = 32
num_epochs = 3
num_workers = 0

# set the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_dir = "/Users/yusuf/Desktop/dataset/"
file_path = f"{base_dir}jpg/files.txt"
save_path = f"{base_dir}dataset.csv"

if __name__ == "__main__":
    
    preprocess(file_path, save_path)

    # Modify this path according to the location of your CSV file
    csv_file_path = save_path

    # Define the ImageNet mean and standard deviation values
    # Replace these values with the actual ImageNet mean and std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Create the transform with ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to a common size (e.g., 224x224)
        transforms.ToTensor(),  # Convert the images to PyTorch tensors
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),  # ImageNet normalization
    ])
    # Create the custom dataset and data loader
    dataset = CustomDataset(csv_file=csv_file_path, transform=transform)


    # split the dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # create train, validation, and test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Example usage:
    num_classes = dataset.num_classes  # Replace with the number of classes in your specific task
    model = PretrainedResNet50(num_classes)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Move the model to the device
    model = model.to(device)
    results = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    test_result = {"test_loss": [], "test_acc": []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        average_loss, accuracy = train(model, train_dataloader, loss_fn, optimizer, device)
        val_average_loss,val_accuracy = validate(model, val_dataloader, loss_fn, device)
        print(f"loss: {average_loss:>7f}, accuracy: {accuracy:>7f}")
        print(f"val_loss: {val_average_loss:>7f}, val_accuracy: {val_accuracy:>7f}")
        results["epoch"].append(epoch)
        results["train_loss"].append(average_loss)
        results["train_acc"].append(accuracy)
        results["val_loss"].append(val_average_loss)
        results["val_acc"].append(val_accuracy)
        df = pd.DataFrame(results)
        df.to_csv(f'Results/history_resnet50.csv', index=False)
        torch.save(model.state_dict(), f'./Models/resnet50.pth')
    test_average_loss,test_accuracy = test(model, test_dataloader, loss_fn, device)
    test_result["test_loss"].append(test_average_loss)
    test_result["test_acc"].append(test_accuracy)
    df = pd.DataFrame(test_result)
    df.to_csv(f'Results/test_resnet50.csv', index=False)

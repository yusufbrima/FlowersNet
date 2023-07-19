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
import pandas as pd
from dataset import CustomDataset
from model import PretrainedResNet50
from utils import preprocess

base_dir = "/Users/yusuf/Desktop/dataset/"
file_path = f"{base_dir}jpg/files.txt"
save_path = f"{base_dir}dataset.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_samples(data_loader, save=False):
    images, labels = next(iter(data_loader))
    
    fig = plt.figure(figsize=(16, 8))
    for i in range(32):
        ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        # convert the one-hot encoded label to an integer
        ax.set_title(np.argmax(labels[i]).numpy().item())
    if save:
        plt.savefig('Figures/plot.png')
    plt.show()
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

    # Create the transform with ImageNet normalization
    tt = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to a common size (e.g., 224x224)
        transforms.ToTensor(),  # Convert the images to PyTorch tensors
    ])


    # Create the custom dataset and data loader
    dataset = CustomDataset(csv_file=csv_file_path, transform=tt)


    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    plot_samples(data_loader, save=True)

    # Load the model and make predictions

    # load model
    num_classes = dataset.num_classes  # Replace with the number of classes in your specific task
    model = PretrainedResNet50(num_classes)

    model_path = "Models/resnet50.pth"
    model.load_state_dict(torch.load(model_path))

    # make prediction for 8 images and plot them
    df = pd.read_csv(csv_file_path)
    fig = plt.figure(figsize=(16, 8))
    for i in range(8):
        idx =  np.random.randint(0, len(df))
        true_label = df.iloc[idx, 1]
        image_path = df.iloc[idx, 0]
        image = Image.open(image_path)
        # transform the image
        t_image = transform(image)
        # add batch dimension
        t_image = t_image.unsqueeze(0)
        image = tt(image)
        with torch.inference_mode():
            pred_label = model(t_image)
            pred_label = pred_label.argmax(dim=1).numpy().item()
        ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
        plt.imshow(image.squeeze(0).permute(1, 2, 0))
        ax.set_title(f"True: {true_label}  Predicted: {pred_label}")
        ax.axis("off")
        # plt.savefig('Figures/prediction_plot.png')
    plt.show()







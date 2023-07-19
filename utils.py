import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image

def preprocess(file_path, save_path='dataset/dataset.csv'):
    """
    Preprocesses a file containing a list of file names and generates a CSV file
    containing two columns: 'files' and 'categories'.

    Args:
        file_path (str): The path to the file containing the list of file names.
        save_path (str, optional): The path to save the generated CSV file. 
            Defaults to 'dataset/dataset.csv'.

    Returns:
        None
    """
    # Open the file in read mode
    with open(file_path, "r") as f:
        # Read all lines and store them in a list
        lines = f.readlines()

    # Remove '\n' from each line and create a list of file names
    file_names = [x.strip() for x in lines]

    # Define the step size for grouping files into categories
    step_size = 80

    base_dir = str(Path(file_path).parent)
    # Initialize the data dictionary to store 'files' and 'categories'
    data = {'files': [], 'categories': []}
    category_label = 0

    for i in range(0, len(file_names), step_size):
        # Group files into chunks of size step_size
        data_files = file_names[i:i + step_size]

        for file in data_files:
            # Append file paths to 'files'
            data['files'].append(f'{base_dir}/' + file)
            # Append the category label to 'categories'
            data['categories'].append(category_label)

        # Increment the category label for the next set of files
        category_label += 1

    # Convert the data dictionary to a DataFrame and save it as a CSV file
    pd.DataFrame(data).to_csv(save_path, index=False)



def train(model, train_loader, criterion, optimizer, device):
    """
    Training function for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimizer (e.g., SGD or Adam) for updating model parameters.
        device (torch.device): The device to which the data and model should be moved (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average training loss for the current epoch.
        float: The accuracy on the training set for the current epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = len(train_loader.dataset)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, labels_indices = torch.max(labels, dim=1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Calculate the number of correct predictions
        _, predicted_labels = torch.max(outputs, 1)
        correct_predictions += (predicted_labels == labels_indices).sum().item()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)} complete. Loss : {loss.item():.4f}")
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy

def validate(model, val_loader, criterion, device):
    """
    Validation function for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        val_loader (torch.utils.data.DataLoader): The DataLoader for validation data.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        device (torch.device): The device to which the data and model should be moved (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average validation loss for the current epoch.
        float: The accuracy on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = len(val_loader.dataset)

    with torch.inference_mode():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, dim=1)
            loss = criterion(outputs, labels_indices)

            total_loss += loss.item() * inputs.size(0)

            # Calculate the number of correct predictions
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels_indices).sum().item()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(val_loader)} complete. Loss : {loss.item():.4f}")

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy

def test(model, test_loader, criterion, device):
    """
    Test function for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The DataLoader for test data.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        device (torch.device): The device to which the data and model should be moved (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average test loss.
        float: The accuracy on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, dim=1)
            loss = criterion(outputs, labels_indices)

            total_loss += loss.item() * inputs.size(0)

            # Calculate the number of correct predictions
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels_indices).sum().item()

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy



if __name__ == "__main__":
    pass
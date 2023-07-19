import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Custom Dataset class for loading data from a CSV file and providing
        one-hot encoded labels.

        Args:
            csv_file (str): Path to the CSV file containing two columns: 
                            'files' (file paths) and 'categories' (labels).
            transform (callable, optional): Optional transformations to apply 
                            to the images. Default is None.

        Attributes:
            data (DataFrame): The DataFrame containing the data from the CSV file.
            CLASSES (list): The list of unique classes present in the dataset.
            num_classes (int): The number of unique classes in the dataset.
            transform (callable): The transformation function applied to the images.
        """
        self.data = pd.read_csv(csv_file)
        self.CLASSES = self.data['categories'].unique().tolist()
        self.num_classes = len(self.CLASSES)
        self.transform = transform

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset based on the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and its one-hot encoded label.
        """
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        # Convert the integer label to one-hot encoding
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1

        return image, one_hot_label

import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the PretrainedResNet50 class.

        Args:
            num_classes (int): Number of output classes for the new classifier layer.
        """
        super(PretrainedResNet50, self).__init__()

        # Load the pre-trained ResNet-50 model from torchvision
        self.resnet50 =  models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all the layers in the model
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Replace the classifier (fully connected) layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Perform a forward pass through the PretrainedResNet50 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) after passing through the model.
        """
        return self.resnet50(x)


if __name__ == "__main__":
    pass

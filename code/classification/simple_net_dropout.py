import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(7,7)),
            nn.BatchNorm2d(10), 
            nn.MaxPool2d(2,1),
            nn.ReLU(),
            nn.Conv2d(10, 15, kernel_size=(7,7)),
            nn.BatchNorm2d(15), 
            nn.MaxPool2d(3,2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(15, 20, kernel_size=(9,9)),
            # nn.BatchNorm2d(20), 
            nn.MaxPool2d(3,3),
            nn.ReLU()
            )

        self.fc_layers = nn.Sequential(
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100,15))

        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) # flatten
        model_output = self.fc_layers(x)
        ############################################################################
        # Student code end
        ############################################################################

        return model_output

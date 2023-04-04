import torch.nn as nn
import torch

class TestModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(TestModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        dummy_image = torch.zeros((1, 3, input_shape[0], input_shape[1]))
        conv_layers_output = self.conv_layers(dummy_image)
        input_size = conv_layers_output.size()[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

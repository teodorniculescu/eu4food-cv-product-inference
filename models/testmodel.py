import torch.nn as nn

class TestModel:
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
            nn.ReLU()
        )
        input_size = 64 * (input_shape[0] // 2**3) * (input_shape[1] // 2**3)
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, num_classes)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (input_shape[0] // 8) * (input_shape[1] // 8), num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

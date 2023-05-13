import torch
from .testmodel import TestModel

AVAILABLE_MODELS = [
    'testmodel',

    'alexnet',

    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',

    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',

    'regnet_y_400mf',
    'regnet_y_800mf',
    'regnet_y_1_6gf',
    'regnet_y_3_2gf',
    'regnet_y_8gf',
    'regnet_x_400mf',
    'regnet_x_800mf',
    'regnet_x_1_6gf',
    'regnet_x_3_2gf',
    'regnet_x_8gf',

    'mobilenet_v3_small'
    'mobilenet_v3_large',
    ]


class ModelFetcher:
    def __init__(self, model_name, image_size, num_classes, device, dropout_prob=0.2):
        self.model_name = model_name
        self.image_size = image_size
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.device = device


    def get_model_and_optimizer_parameters(self):
        optimization_param = None

        if self.model_name == 'testmodel':
            self.model = TestModel(self.image_size, self.num_classes)
        else:
            self.model = torch.hub.load('pytorch/vision:v0.11.3', self.model_name, pretrained=True)
        if self.model_name == 'testmodel':
            optimization_param = self.model.parameters()

        elif self.model_name[:3] == 'vgg' or self.model_name[:12] == 'efficientnet' or self.model_name == 'alexnet': 
            for param in self.model.parameters():
                param.requires_grad = False

            linear_layer_position = len(self.model.classifier) - 1
            in_features = self.model.classifier[linear_layer_position].in_features
            self.model.classifier[linear_layer_position] = torch.nn.Sequential(
                    torch.nn.Linear(in_features, self.num_classes),
                    torch.nn.Softmax(dim=1)
                    )

            optimization_param = self.model.classifier[linear_layer_position].parameters()

        elif self.model_name[:6] == 'resnet' or self.model_name[:6] == 'regnet':
            for param in self.model.parameters():
                param.requires_grad = False
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features, self.num_classes),
                torch.nn.Softmax(dim=1)
                )

            optimization_param = self.model.fc.parameters()

        return self.model, optimization_param

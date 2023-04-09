from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np
import os
import random
from cutout_transform import CutoutNumpy
import cv2

class CustomDataset(Dataset):
    def __init__(self, root_dir, image_size=(32, 32), mean=(0.5, 0.5, 0.5),  std=(0.5, 0.5, 0.5)):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.dataset = ImageFolder(root_dir, transform=self.transform)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def __len__(self):
        return len(self.dataset)

    def num_classes(self):
        # Return the number of unique labels in the dataset
        return len(self.dataset.classes)

    def get_classes(self):
        # Return the number of unique labels in the dataset
        return self.dataset.classes

    def get_loaders(self, batch_size, num_workers, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise Exception("ERROR: Train, validation and test ratios must sum to 1.")

        train_size = int(train_ratio * len(self))
        val_size = int(val_ratio * len(self))
        test_size = len(self) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(self, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

def get_loader_class_count(loader, num_classes):
    total_class_count = {number: 0 for number in range(num_classes)}
    for _, labels in tqdm(loader):
        labels = labels.cpu().detach().numpy()
        unique_numbers, counts = np.unique(labels, return_counts=True)
        for number, count in zip(unique_numbers, counts):
            total_class_count[number] += count
    return total_class_count

class CustomImagePathDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = self.transform(image)
        return image, label

class CustomClassBalancedDataset(Dataset):
    def __init__(self, root_dir, image_size=(32, 32), mean=(0.5, 0.5, 0.5),  std=(0.5, 0.5, 0.5), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, augment=True):
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise Exception("ERROR: Train, validation and test ratios must sum to 1.")

        self.image_size = image_size
        self.mean = mean
        self.std = std

        if augment:
            cutout_size = int(image_size[0] * 0.01)
            print('augment with cutout size of', cutout_size)
            self.train_transform = transforms.Compose([
                #CutoutNumpy(cutout_percent=0.08, probability=1), 
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(image_size, (0.7, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        else:
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.classes = os.listdir(root_dir)
        self.classes.sort()

        train_image_paths, val_image_paths, test_image_paths = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        for class_idx, class_name in enumerate(self.classes):
            image_paths = self.__get_image_paths__(os.path.join(root_dir, class_name))

            data_size = len(image_paths)
            train_size = int(train_ratio * data_size)
            val_size = int(val_ratio * data_size)
            test_size = data_size - train_size - val_size

            if test_size < 0:
                print(f'WARNING: Class {class_name} will not have test test images due to the small size of the dataset of the split. Currently there are {train_size} train images and {val_size} val images out of {data_size} total images.')

            train_image_paths += image_paths[:train_size]
            val_image_paths += image_paths[train_size:train_size + val_size]
            test_image_paths += image_paths[train_size + val_size:]

            train_labels += [class_idx] * train_size
            val_labels += [class_idx] * val_size
            test_labels += [class_idx] * test_size

        self.train_dataset = CustomImagePathDataset(train_image_paths, train_labels, self.train_transform)
        self.val_dataset = CustomImagePathDataset(val_image_paths, val_labels, self.transform)
        self.test_dataset = CustomImagePathDataset(test_image_paths, test_labels, self.transform)

    def __get_image_paths__(self, root_dir, extensions=('.jpg', '.png', '.jpeg'), shuffle=True):
        image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file_path in files:
                if file_path.lower().endswith(extensions):
                    image_path = os.path.join(root, file_path)
                    image_paths.append(image_path)

        if shuffle:
            random.shuffle(image_paths)

        return image_paths

    def num_classes(self):
        # Return the number of unique labels in the dataset
        return len(self.classes)

    def get_classes(self):
        # Return the number of unique labels in the dataset
        return self.classes

    def get_loaders(self, batch_size, num_workers):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

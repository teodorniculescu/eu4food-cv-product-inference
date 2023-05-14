from torchvision.datasets import ImageFolder
import time
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

def get_transform(augment_type, image_size, mean, std, use_mean_std):
    transforms_list = [
        transforms.ToPILImage(),
    ]
    if augment_type is None or augment_type == 'None':
        pass

    elif augment_type == 'Custom':
        transforms_list += [
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.7, 2)),
        ]

    elif augment_type == 'RandAugment':
        transforms_list += [
            transforms.RandAugment(),
        ]

    elif augment_type == 'AugMix':
        transforms_list += [
            transforms.AugMix(),
        ]

    elif augment_type == 'TrivialAugmentWide':
        transforms_list += [
            transforms.TrivialAugmentWide(),
        ]

    else:
        raise Exception(f"ERROR: Unknown augment type {augment_type}")

    transforms_list += [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]

    if use_mean_std:
        transforms_list += [
            transforms.Normalize(mean=mean, std=std)
        ]

    else:
        transforms_list += [
            transforms.Lambda(lambda x: x * 255),
        ]
    
    return transforms.Compose(transforms_list)


class CustomImagePathDataset(Dataset):
    def __init__(self, name, image_paths, images, preload_images, labels, image_size, mean, std, use_mean_std, augment_type=None):
        self.name = name
        self.image_paths = image_paths
        self.images = images
        self.preload_images = preload_images
        self.labels = labels
        self.use_mean_std = use_mean_std
        self.transform = get_transform(augment_type, image_size, mean, std, use_mean_std)
        self.elapsed_time_list = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        start_time = time.time()
        label = self.labels[idx]
        if self.preload_images:
            #st = time.time()
            image = self.images[idx]
            #et_get = time.time() - st

        else:
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)

        #st = time.time()
        image = self.transform(image)
        #et_tr = time.time() - st

        elapsed_time = time.time() - start_time
        #print(elapsed_time,
                #'get', round(et_get/elapsed_time, 3),
                #'tr', round(et_tr/elapsed_time, 3),
                #)
        self.elapsed_time_list.append(elapsed_time)
        return image, label

class CustomClassBalancedDataset(Dataset):
    def __init__(self, root_dir, preload_images=False, image_size=(32, 32), mean=(0.5, 0.5, 0.5),  std=(0.5, 0.5, 0.5), use_mean_std=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, augment_train=None, augment_valid=None):
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise Exception("ERROR: Train, validation and test ratios must sum to 1.")

        self.classes = os.listdir(root_dir)
        self.classes.sort()

        self.excluded_classes = []
        for class_name in self.classes:
            image_paths = self.__get_image_paths__(os.path.join(root_dir, class_name))

            data_size = len(image_paths)
            train_size = int(train_ratio * data_size)
            val_size = int(val_ratio * data_size)
            test_size = data_size - train_size - val_size

            if test_size == 0 or val_size == 0 or train_size == 0:
                print(f'WARNING: Class {class_name} will not be used due to not enough images in acording to current split. Class has {data_size} images in total split as {train_size} {val_size} {test_size}')
                self.excluded_classes.append(class_name)

        for class_name in self.excluded_classes:
            self.classes.remove(class_name)

        if len(self.classes) == 0:
            raise Exception('ERROR: No classes left in the dataloader')

        print('classes', self.classes)

        train_image_paths, val_image_paths, test_image_paths = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        for class_idx, class_name in enumerate(self.classes):
            image_paths = self.__get_image_paths__(os.path.join(root_dir, class_name))

            data_size = len(image_paths)
            train_size = int(train_ratio * data_size)
            val_size = int(val_ratio * data_size)
            test_size = data_size - train_size - val_size

            print(f'Class {class_name} split {train_size} {val_size} {test_size}')

            train_image_paths += image_paths[:train_size]
            val_image_paths += image_paths[train_size:train_size + val_size]
            test_image_paths += image_paths[train_size + val_size:]

            train_labels += [class_idx] * train_size
            val_labels += [class_idx] * val_size
            test_labels += [class_idx] * test_size

        if preload_images:
            train_images = self.load_images_from_path(train_image_paths)
            val_images = self.load_images_from_path(val_image_paths)
            test_images = self.load_images_from_path(test_image_paths)

        else:
            train_images, val_images, test_images = None, None, None

        self.train_dataset = CustomImagePathDataset('train', train_image_paths, train_images, preload_images, train_labels, image_size, mean, std, use_mean_std, augment_type=augment_train)
        self.val_dataset = CustomImagePathDataset('val', val_image_paths, val_images, preload_images, val_labels, image_size, mean, std, use_mean_std, augment_type=augment_valid)
        self.test_dataset = CustomImagePathDataset('test', test_image_paths, test_images, preload_images, test_labels, image_size, mean, std, use_mean_std)

    def load_images_from_path(self, image_paths):
        images = []
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            images.append(image)

        return images

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

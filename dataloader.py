from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

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


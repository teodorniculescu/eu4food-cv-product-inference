from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset

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

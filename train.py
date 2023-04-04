from models import TestModel
from dataloader import CustomDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score
import os

def get_args():
    # Set up the argparse object
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate to use for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum to use for optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Path to save the trained model')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Size of the input images as a tuple (height, width)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')

    # Parse the arguments and call the training function
    args = parser.parse_args()

    args.image_size = tuple(args.image_size)

    return args


def train_validate_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, save_path):
    scores = {'train': [], 'val': []}
    best_f1 = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            f1 = []
            acc = []

            for images, labels in tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                f1.append(f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted'))
                acc.append(accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()))

            epoch_loss = running_loss / len(loader.dataset)
            epoch_f1 = sum(f1) / len(f1)
            epoch_acc = sum(acc) / len(acc)

            scores[phase].append({'epoch': epoch, 'loss': epoch_loss, 'f1': epoch_f1, 'accuracy': epoch_acc})
            print(phase, scores[phase][-1])

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), save_path)

    return scores


def test_model(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    f1 = []
    acc = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            f1.append(f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted'))
            acc.append(accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()))

    test_loss = running_loss / len(test_loader.dataset)
    test_f1 = sum(f1) / len(f1)
    test_acc = sum(acc) / len(acc)

    test_scores = {'loss': test_loss, 'f1': test_f1, 'accuracy': test_acc}
    print('test', test_scores)
    return test_scores


def main():
    args = get_args()

    dataset = CustomDataset(args.data_dir, image_size=args.image_size)
    args.num_classes = dataset.num_classes()
    args.classes = dataset.get_classes()

    model = TestModel(args.image_size, args.num_classes).to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    scores = train_validate_model(model, train_loader, val_loader, optimizer, criterion, args.device, args.num_epochs, args.save_path)
    test_scores = test_model(model, test_loader, criterion, args.device)

    scores['test'] = test_scores

    with open('scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

    with open('args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    main()

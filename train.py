from datetime import datetime
from models import ModelFetcher, AVAILABLE_MODELS
from dataloader import CustomDataset, get_loader_class_count, CustomClassBalancedDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_args():
    # Set up the argparse object
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=AVAILABLE_MODELS, help='The type of the model used in the training process')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate to use for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum to use for optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--model_name', type=str, default='best_model.pth', help='Name of the best model')
    parser.add_argument('--save_path', type=str, default='train_results', help='Path to save the trained model')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Size of the input images as a tuple (height, width)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for loading the dataset')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406], help='Mean used for normalizing the images')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225], help='Std used for normalizing the images')
    parser.add_argument('--show_class_count', action='store_true')
    parser.add_argument('--imbalanced_dataset', action='store_true')

    # Parse the arguments and call the training function
    args = parser.parse_args()

    args.image_size = tuple(args.image_size)
    args.norm_mean = tuple(args.norm_mean)
    args.norm_std = tuple(args.norm_std)

    return args


def train_validate_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, model_save_path):
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
                torch.save(model.state_dict(), model_save_path)
                print('saved model')

    return scores


def test_model(model, best_model_save_path, test_loader, criterion, device, conf_matrix_save_path, class_labels):
    model.load_state_dict(torch.load(best_model_save_path))
    model.eval()

    running_loss = 0.0
    f1 = []
    acc = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            f1.append(f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted'))
            acc.append(accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_f1 = sum(f1) / len(f1)
    test_acc = sum(acc) / len(acc)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_labels))))
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(conf_matrix_save_path, bbox_inches='tight')

    test_scores = {'loss': test_loss, 'f1': test_f1, 'accuracy': test_acc}
    print('test', test_scores)
    return test_scores

def get_current_timestamp():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def main():
    args = get_args()
    args.timestamp = get_current_timestamp()
    args.save_path = os.path.join(args.save_path, args.timestamp)
    torch_device = torch.device(args.device)

    os.makedirs(os.path.join(args.save_path))

    if args.imbalanced_dataset:
        dataset = CustomDataset(args.data_dir, image_size=args.image_size, mean=args.norm_mean, std=args.norm_std)

    else:
        print('Using class balanced dataset')
        dataset = CustomClassBalancedDataset(args.data_dir, image_size=args.image_size, mean=args.norm_mean, std=args.norm_std)

    train_loader, val_loader, test_loader = dataset.get_loaders(args.batch_size, args.num_workers)
    num_classes = dataset.num_classes()
    classes = dataset.get_classes()

    if args.show_class_count:
        print('train dataset image count', get_loader_class_count(train_loader, num_classes))
        print('val dataset image count', get_loader_class_count(val_loader, num_classes))
        print('test dataset image count', get_loader_class_count(test_loader, num_classes))

    model_fetcher = ModelFetcher(args.model, args.image_size, num_classes, torch_device)

    model, optimizer_param = model_fetcher.get_model_and_optimizer_parameters()
    model.to(torch_device)

    optimizer = optim.SGD(optimizer_param, lr=args.learning_rate, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    model_save_path = os.path.join(args.save_path, args.model_name)
    scores = train_validate_model(model, train_loader, val_loader, optimizer, criterion, torch_device, args.num_epochs, model_save_path)

    conf_matrix_save_path = os.path.join(args.save_path, 'confusion_matrix.png')
    test_scores = test_model(model, model_save_path, test_loader, criterion, torch_device, conf_matrix_save_path, classes)

    scores['test'] = test_scores

    scores_save_path = os.path.join(args.save_path, 'scores.json')
    with open(scores_save_path, 'w') as f:
        json.dump(scores, f, indent=4)

    args_save_path = os.path.join(args.save_path, 'args.json')
    with open(args_save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    obj_names_path = os.path.join(args.save_path, 'obj.names')
    with open(obj_names_path, 'w') as f:
        for class_name in classes:
            f.write(class_name + '\n')

if __name__ == '__main__':
    main()

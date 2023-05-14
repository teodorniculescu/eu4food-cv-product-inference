from datetime import datetime
from models import ModelFetcher, AVAILABLE_MODELS
from dataloader import CustomDataset, get_loader_class_count, CustomClassBalancedDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import numpy as np


def get_args():
    # Set up the argparse object
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=AVAILABLE_MODELS, help='The type of the model used in the training process')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate to use for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum to use for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay to use for optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--model_name', type=str, default='best_model.pth', help='Name of the best model')
    parser.add_argument('--save_path', type=str, default='train_results', help='Path to save the trained model')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Size of the input images as a tuple (height, width)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for loading the dataset')

    parser.add_argument('--use_mean_std', action='store_true')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406], help='Mean used for normalizing the images')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225], help='Std used for normalizing the images')
    parser.add_argument('--ratio', type=float, nargs=3, default=[0.5, 0.25, 0.25], help='Train-Validation-Test data ratio')
    parser.add_argument('--show_class_count', action='store_true')
    augment_choices = ('None', 'RandAugment', 'AugMix', 'TrivialAugmentWide', 'Custom')
    parser.add_argument('--augment_train', type=str, default='None', choices=augment_choices)
    parser.add_argument('--augment_valid', type=str, default='None', choices=augment_choices)

    parser.add_argument('--preload_images', action='store_true')
    parser.add_argument('--dont_save_model', action='store_true')
    

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
            epoch_start_time = time.time()
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            f1 = []
            acc = []

            for images, labels in tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}", leave=False):
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

            epoch_loss = round(epoch_loss, 4)
            epoch_f1 = round(epoch_f1, 4)
            epoch_acc = round(epoch_acc, 4)
            elapsed_time = round(time.time() - epoch_start_time, 4)

            scores[phase].append({'epoch': epoch, 'loss': epoch_loss, 'f1': epoch_f1, 'accuracy': epoch_acc, 'time': elapsed_time})

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), model_save_path)
                print('saved model')
                
        print('train', scores['train'][-1], 'val', scores['val'][-1], )

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
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

def draw_plot(scores, plot_save_path):
    train_loss = [epoch['loss'] for epoch in scores['train']]
    val_loss = [epoch['loss'] for epoch in scores['val']]
    train_acc = [epoch['accuracy'] for epoch in scores['train']]
    val_acc = [epoch['accuracy'] for epoch in scores['val']]
    train_f1 = [epoch['f1'] for epoch in scores['train']]
    val_f1 = [epoch['f1'] for epoch in scores['val']]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

    ax1.plot(train_loss, label='train')
    ax1.plot(val_loss, label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_acc, label='train')
    ax2.plot(val_acc, label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    ax3.plot(train_f1, label='train')
    ax3.plot(val_f1, label='val')
    ax3.set_title('F1 score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 score')
    ax3.legend()

    fig.savefig(plot_save_path)


def main():
    args = get_args()
    args.timestamp = get_current_timestamp()
    save_path = os.path.join(args.save_path, args.timestamp)
    torch_device = torch.device(args.device)

    os.makedirs(os.path.join(save_path))

    dataset = CustomClassBalancedDataset(
            args.data_dir, 
            preload_images=args.preload_images,
            image_size=args.image_size,
            mean=args.norm_mean, std=args.norm_std, use_mean_std=args.use_mean_std,
            train_ratio=args.ratio[0], val_ratio=args.ratio[1], test_ratio=args.ratio[2],
            augment_train=args.augment_train, augment_valid=args.augment_valid
    )

    train_loader, val_loader, test_loader = dataset.get_loaders(args.batch_size, args.num_workers)
    num_classes = dataset.num_classes()
    classes = dataset.get_classes()

    if args.show_class_count:
        print('Generating class count for datasets')
        print('train dataset image count', get_loader_class_count(train_loader, num_classes))
        print('val dataset image count', get_loader_class_count(val_loader, num_classes))
        print('test dataset image count', get_loader_class_count(test_loader, num_classes))

    model_fetcher = ModelFetcher(args.model, args.image_size, num_classes, torch_device)

    model, optimizer_param = model_fetcher.get_model_and_optimizer_parameters()
    model.to(torch_device)

    optimizer = optim.SGD(optimizer_param, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    model_save_path = os.path.join(save_path, args.model_name)
    scores = train_validate_model(model, train_loader, val_loader, optimizer, criterion, torch_device, args.num_epochs, model_save_path)
    train_val_elapsed_time = time.time() - start_time

    start_time = time.time()
    conf_matrix_save_path = os.path.join(save_path, 'confusion_matrix.png')
    test_scores = test_model(model, model_save_path, test_loader, criterion, torch_device, conf_matrix_save_path, classes)
    test_elapsed_time = time.time() - start_time

    if args.dont_save_model:
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    scores['test'] = test_scores

    args.elapsed_time = {}
    args.elapsed_time['dataset'] = {}
    for ds in [dataset.train_dataset, dataset.val_dataset, dataset.test_dataset]:
        l = np.array(ds.elapsed_time_list)
        mean = np.mean(l)
        std = np.std(l)
        args.elapsed_time['dataset'][ds.name] = {'mean': mean, 'std': std}
    args.elapsed_time['dataset']['train_val'] = train_val_elapsed_time
    args.elapsed_time['dataset']['test'] = test_elapsed_time

    args.dataset_image_count = {'used': dataset.split_image_count, 'excluded': dataset.excluded_split_image_count}


    plot_save_path = os.path.join(save_path, 'train_plot.png')
    draw_plot(scores, plot_save_path)

    scores_save_path = os.path.join(save_path, 'scores.json')
    with open(scores_save_path, 'w') as f:
        json.dump(scores, f, indent=4)

    args_save_path = os.path.join(save_path, 'args.json')
    with open(args_save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    obj_names_path = os.path.join(save_path, 'obj.names')
    with open(obj_names_path, 'w') as f:
        for class_name in classes:
            f.write(class_name + '\n')

    excluded_obj_names_path = os.path.join(save_path, 'excluded.obj.names')
    with open(excluded_obj_names_path, 'w') as f:
        for class_name in dataset.excluded_classes:
            f.write(class_name + '\n')

if __name__ == '__main__':
    main()

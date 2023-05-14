import argparse
from dataloader import get_loader_class_count, CustomClassBalancedDataset
import cv2
import numpy as np 


def main():
    args = get_args()
    dataset = CustomClassBalancedDataset(
            args.data_dir, 
            preload_images=args.preload_images,
            image_size=args.image_size,
            mean=args.norm_mean, std=args.norm_std, use_mean_std=args.use_mean_std,
            train_ratio=args.ratio[0], val_ratio=args.ratio[1], test_ratio=args.ratio[2],
            augment_train=args.augment_train, augment_valid=args.augment_valid
            )
    train_loader, val_loader, test_loader = dataset.get_loaders(1, 0)

    for images, _ in train_loader:
        frame = images.numpy()[0]
        frame = np.transpose(frame, (1, 2, 0))
        frame = frame.astype(np.uint8)
        cv2.imshow('adsf', frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()
    

def get_args():
    # Set up the argparse object
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Size of the input images as a tuple (height, width)')
    parser.add_argument('--use_mean_std', action='store_true')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406], help='Mean used for normalizing the images')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225], help='Std used for normalizing the images')
    parser.add_argument('--ratio', type=float, nargs=3, default=[0.5, 0.25, 0.25], help='Train-Validation-Test data ratio')
    augment_choices = ('RandAugment', 'AugMix', 'TrivialAugmentWide', 'Custom')
    parser.add_argument('--augment_train', type=str, default=None, choices=augment_choices)
    parser.add_argument('--augment_valid', type=str, default=None, choices=augment_choices)

    parser.add_argument('--preload_images', action='store_true')

    args = parser.parse_args()

    args.image_size = tuple(args.image_size)
    args.norm_mean = tuple(args.norm_mean)
    args.norm_std = tuple(args.norm_std)

    return args

if __name__ == '__main__':
    main()


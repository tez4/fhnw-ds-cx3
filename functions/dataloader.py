import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, size, config, augmentation):

        self.image_paths = image_paths
        self.size = size
        self.augmentation = augmentation
        self.random_horizontal_flip = config["random_horizontal_flip"]

        transformations = []

        transformations.append(transforms.Resize((self.size, self.size)))

        if augmentation:
            if self.random_horizontal_flip:
                transformations.append(transforms.RandomHorizontalFlip(p=0.5))

        transformations.append(transforms.ToTensor())

        self.transform = transforms.Compose(transformations)

    def __len__(self):
        return len(self.image_paths)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        input_image = Image.open(f"{self.image_paths[idx]}/input.png")
        output_image = Image.open(f"{self.image_paths[idx]}/distance.png")

        seed = random.randint(0, 2**32)

        if self.transform:
            self._set_seed(seed)
            input_image = self.transform(input_image)
            self._set_seed(seed)
            output_image = self.transform(output_image)

        return input_image, output_image


def get_data_loaders(config, shuffle):

    size = config["image_size"]
    batch_train_size = config["train_batch_size"]
    batch_val_size = config["val_batch_size"]
    batch_test_size = config["test_batch_size"]
    data_path = f"./input/{config['dataset']}"
    num_workers = config["num_workers"]

    main_train_dir = f"{data_path}/training"
    train_dirs = [f'{main_train_dir}/{d}' for d in os.listdir(main_train_dir) if os.path.isdir(f'{main_train_dir}/{d}')]
    train_dataset = SegmentationDataset(
        image_paths=train_dirs,
        augmentation=True,
        size=size,
        config=config,
    )

    main_val_dir = f"{data_path}/validation"
    val_dirs = [f'{main_val_dir}/{d}' for d in os.listdir(main_val_dir) if os.path.isdir(f'{main_val_dir}/{d}')]
    val_dataset = SegmentationDataset(
        image_paths=val_dirs,
        augmentation=False,
        size=size,
        config=config
    )

    main_test_dir = f"{data_path}/test"
    test_dirs = [f'{main_test_dir}/{d}' for d in os.listdir(main_test_dir) if os.path.isdir(f'{main_test_dir}/{d}')]
    test_dataset = SegmentationDataset(
        image_paths=test_dirs,
        augmentation=False,
        size=size,
        config=config
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_train_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_val_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_test_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def show_tensor(tensor, element=None, multiplier=255):

    if element is None:
        image_tensor = tensor
    else:
        image_tensor = tensor[element]

    if not isinstance(image_tensor, np.ndarray):
        image_tensor = image_tensor.detach().numpy()

    image_tensor = (image_tensor * multiplier).astype("uint8")

    if image_tensor.shape[0] != 1:
        image_tensor = np.transpose(image_tensor, (1, 2, 0))
        Image.fromarray(image_tensor).show()
    else:
        Image.fromarray(np.squeeze(image_tensor), mode="L").show()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders(
        {
            "image_size": 512,
            "train_batch_size": 8,
            "val_batch_size": 8,
            "test_batch_size": 8,
            "random_horizontal_flip": True,
            "dataset": "experiment_95_preprocessed",
        },
        shuffle=True,
        num_workers=0
    )

    train_iter = iter(train_loader)
    input_image, output_image = next(train_iter)
    show_tensor(output_image, element=0)

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
        self.color_jitter = config["color_jitter"]
        self.random_horizontal_flip = config["random_horizontal_flip"]

        transformations_input = []
        transformations_output = []

        transformations_input.append(transforms.Resize((self.size, self.size)))
        transformations_output.append(transforms.Resize((self.size, self.size)))

        if self.augmentation:
            if self.random_horizontal_flip:
                transformations_input.append(transforms.RandomHorizontalFlip(p=0.5))
                transformations_output.append(transforms.RandomHorizontalFlip(p=0.5))

            if self.color_jitter:
                transformations_input.append(
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                )

        transformations_input.append(transforms.ToTensor())
        transformations_output.append(transforms.ToTensor())

        self.transform_input = transforms.Compose(transformations_input)
        self.transform_output = transforms.Compose(transformations_output)

    def __len__(self):
        return len(self.image_paths)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        input_image = Image.open(f"{self.image_paths[idx]}/input.png")
        distance_image = Image.open(f"{self.image_paths[idx]}/distance.png")
        mask_image = Image.open(f"{self.image_paths[idx]}/mask.png")
        normals_image = Image.open(f"{self.image_paths[idx]}/normals.png")
        output_1_image = Image.open(f"{self.image_paths[idx]}/output_1.png")

        seed = random.randint(0, 2**32)

        self._set_seed(seed)
        input_array = self.transform_input(input_image)
        self._set_seed(seed)
        distance_array = self.transform_output(distance_image)
        self._set_seed(seed)
        mask_array = self.transform_output(mask_image)
        self._set_seed(seed)
        normals_array = self.transform_output(normals_image)
        self._set_seed(seed)
        output_1_array = self.transform_output(output_1_image)

        output_array = torch.cat((distance_array, mask_array, normals_array, output_1_array), dim=0)

        return input_array, output_array, self.image_paths[idx]


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

    if isinstance(num_workers, str):
        if num_workers == "auto":
            num_workers = os.cpu_count()
        else:
            raise NotImplementedError()

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

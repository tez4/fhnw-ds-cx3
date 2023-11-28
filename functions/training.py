import os
import wandb
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from models import UNet, BaseNet
from pathlib import Path
from datetime import datetime
from scores import mean_pixel_distance
from tables import create_examples_tables, get_video_arrays, create_video_tables


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    def __init__(
            self, config: dict, wandb_config: dict, model, discriminator, optimizer, optimizer_discriminator,
            criterion, criterion_discriminator, train_loader, val_loader):

        self.config = config
        self.wandb_config = wandb_config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.discriminator = discriminator
        self.optimizer_discriminator = optimizer_discriminator
        self.criterion_discriminator = criterion_discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.num_epochs_without_improvement = 0
        self.video_arrays = None

        # find optimizer name
        if isinstance(optimizer, torch.optim.SGD):
            optimizer_name = 'SGD'
        elif isinstance(optimizer, torch.optim.Adam):
            optimizer_name = 'Adam'
        else:
            raise ValueError('unknown optimizer')

        # move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        elif torch.cuda.device_count() <= 1 and isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        self.model.to(self.device)
        self.discriminator.to(self.device)
        self.criterion.to(self.device)
        print(f'Selected device: {self.device}')

        # Initialize wandb
        config['device'] = self.device
        config['optimizer'] = optimizer_name

        wandb.init(
            project=wandb_config['project'],
            name=f"{config['name']}-{config['epochs']}-epochs-{config['start_time']}",
            entity=wandb_config['entity'],
            group=wandb_config['group'],
            tags=wandb_config["tags"] + (['test-batch'] if config['is_test_batch'] else []),
            config=config
        )

        # start training loop
        wandb.watch(model)

    def train(self, step: int):
        self.model.train()
        self.discriminator.train()
        running_train_loss = []
        running_train_acc = []
        for i, (inputs, targets, _) in enumerate(self.train_loader):
            if self.config["is_test_batch"] and i > 1:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)

            if self.config['has_discriminator']:
                self.optimizer_discriminator.zero_grad()

                real_output = self.discriminator(inputs, targets)
                loss_discriminator_real = self.criterion_discriminator(real_output, torch.ones_like(real_output))

                fake_output = self.discriminator(inputs, outputs.detach())
                loss_discriminator_fake = self.criterion_discriminator(fake_output, torch.zeros_like(fake_output))

                loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                loss += loss_discriminator.detach() * self.config['loss_lambda']

            loss.backward()
            self.optimizer.step()

            running_train_loss.append(loss.item())
            running_train_acc.append(mean_pixel_distance(outputs, targets).mean().item())

        self.train_loss = np.mean(running_train_loss)
        self.train_acc = np.mean(running_train_acc)

        print(f"Trained epoch {step + 1}/{self.config['epochs']}")

    def validate(self, step: int):
        self.model.eval()
        running_val_loss = []
        running_val_acc = []
        with torch.no_grad():

            # log_images(model, config, test_loader, device, step + 1, epoch_confidence_images)

            for i, (inputs, targets, _) in enumerate(self.val_loader):
                if self.config["is_test_batch"] and i > 1:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_val_loss.append(loss.item())
                running_val_acc.append(mean_pixel_distance(outputs, targets).mean().item())

        self.val_loss = np.mean(running_val_loss)
        self.val_acc = np.mean(running_val_acc)

        if self.val_loss < self.best_val_loss:
            self.best_val_loss = self.val_loss
            self.num_epochs_without_improvement = 0
            self.best_epoch = step + 1
        else:
            self.num_epochs_without_improvement += 1

        if self.val_acc > self.best_val_acc:
            self.best_val_acc = self.val_acc

    def log_values(self, step: int):
        wandb.log({
            "epoch": step + 1,
            "Loss/a_train_loss": self.train_loss,
            "Loss/b_val_loss": self.val_loss,
            "Acc/a_train_acc": self.train_acc,
            "Acc/b_val_acc": self.val_acc,
        })

        self.video_arrays = get_video_arrays(
            self.video_arrays, self.model, self.val_loader, self.device, self.wandb_config['video_images']
        )

        if step + 1 in self.wandb_config['examples_epochs']:
            create_examples_tables(
                self.model, self.val_loader, self.device, step + 1, self.wandb_config['table_images'],
                f'Examples/Validation Examples Epoch {step + 1}'
            )

        print(f"Validated epoch {step + 1}/{self.config['epochs']}")
        print(f"Acc: {round(self.train_acc, 5)}, Validation Acc: {round(self.val_acc, 5)}")

    def finish_training(self, step):
        wandb.log({
            "Loss/c_best_val_loss": self.best_val_loss,
            "Acc/c_best_val_acc": self.best_val_acc,
        })

        create_video_tables(self.video_arrays, 'Videos/Validation Video Examples')

        create_examples_tables(
            self.model, self.val_loader, self.device, step + 1, self.wandb_config['table_images'],
            'Examples/Finished Validation Examples'
        )

        wandb.finish()

        print(f"FINISHED! Best epoch: {self.best_epoch}, Best Accuracy: {self.best_val_acc}")

    def save_model(self, path: str = './models/'):
        self.model.eval()
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model, f'{path}model_{self.config["name"]}_{self.config["start_time"]}.pth')
        print('Model saved!')

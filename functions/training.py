import os
import wandb
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from scores import mean_squared_error, image_mean_squared_error
from tables import create_examples_tables, create_real_tables, get_video_arrays, create_video_tables


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
            criterion, criterion_discriminator, train_loader, val_loader, real_loader):

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
        self.real_loader = real_loader

        self.best_val_mse = 0.0
        self.best_val_image_mse = 0.0
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
        running_train_mse = []
        running_train_image_mse = []
        for i, (inputs, targets, _) in enumerate(self.train_loader):
            if self.config["is_test_batch"] and i > 1:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
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
            running_train_mse.append(mean_squared_error(outputs, targets).mean().item())
            running_train_image_mse.append(image_mean_squared_error(outputs, targets).mean().item())

        self.train_loss = np.mean(running_train_loss)
        self.train_mse = np.mean(running_train_mse)
        self.train_image_mse = np.mean(running_train_image_mse)

        print(f"Trained epoch {step + 1}/{self.config['epochs']}")

    def validate(self, step: int):
        self.model.eval()
        running_val_loss = []
        running_val_mse = []
        running_val_image_mse = []
        with torch.no_grad():

            # log_images(model, config, test_loader, device, step + 1, epoch_confidence_images)

            for i, (inputs, targets, _) in enumerate(self.val_loader):
                if self.config["is_test_batch"] and i > 1:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_val_loss.append(loss.item())
                running_val_mse.append(mean_squared_error(outputs, targets).mean().item())
                running_val_image_mse.append(image_mean_squared_error(outputs, targets).mean().item())

        self.val_loss = np.mean(running_val_loss)
        self.val_mse = np.mean(running_val_mse)
        self.val_image_mse = np.mean(running_val_image_mse)

        if self.val_loss < self.best_val_loss:
            self.best_val_loss = self.val_loss
            self.num_epochs_without_improvement = 0
            self.best_epoch = step + 1
        else:
            self.num_epochs_without_improvement += 1

        if self.val_mse > self.best_val_mse:
            self.best_val_mse = self.val_mse

        if self.val_image_mse > self.best_val_image_mse:
            self.best_val_image_mse = self.val_image_mse

    def log_values(self, step: int):
        wandb.log({
            "epoch": step + 1,
            "Loss/a_train_loss": self.train_loss,
            "Loss/b_val_loss": self.val_loss,
            "MSE/a_train_mse": self.train_mse,
            "MSE/b_val_mse": self.val_mse,
            "Image MSE/a_train_image_mse": self.train_image_mse,
            "Image MSE/b_val_image_mse": self.val_image_mse,
        })

        self.video_arrays = get_video_arrays(
            self.video_arrays, self.model, self.val_loader, self.device, self.wandb_config['video_images']
        )

        if step + 1 in self.wandb_config['examples_epochs']:
            create_examples_tables(
                self.model, self.val_loader, self.device, step + 1, self.wandb_config['table_images'],
                f'Examples/Validation Examples Epoch {step + 1}'
            )

            create_real_tables(
                self.model, self.real_loader, self.device, step + 1,
                f'Examples/Real Examples Epoch {step + 1}'
            )

        print(f"Validated epoch {step + 1}/{self.config['epochs']}")
        print(f"MSE: {round(self.train_mse, 5)}, Validation MSE: {round(self.val_mse, 5)}")

    def finish_training(self, step):
        wandb.log({
            "Loss/c_best_val_loss": self.best_val_loss,
            "MSE/c_best_val_mse": self.best_val_mse,
            "Image MSE/c_best_val_image_mse": self.best_val_image_mse,
        })

        create_video_tables(self.video_arrays, 'Videos/Validation Video Examples')

        create_examples_tables(
            self.model, self.val_loader, self.device, step + 1, self.wandb_config['table_images'],
            'Finished Examples/Finished Validation Examples'
        )

        create_real_tables(
            self.model, self.real_loader, self.device, step + 1,
            'Finished Examples/Finished Real Examples'
        )

        wandb.finish()

        print(f"FINISHED! Best epoch: {self.best_epoch}, Best MSE: {self.best_val_mse}")

    def save_model(self, path: str = './models/'):
        self.model.eval()
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model, f'{path}model_{self.config["name"]}_{self.config["start_time"]}.pth')
        print('Model saved!')

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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train_model(model, optimizer, criterion, config, wandb_config, train_loader, val_loader):

    # find optimizer name
    if isinstance(optimizer, torch.optim.SGD):
        optimizer_name = 'SGD'
    elif isinstance(optimizer, torch.optim.Adam):
        optimizer_name = 'Adam'
    else:
        raise ValueError('unknown optimizer')

    # move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    print(f'Selected device: {device}')

    # Initialize wandb
    config['device'] = device
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
    best_val_acc = 0.0
    best_epoch = 0
    num_epochs_without_improvement = 0
    loop = range(config["epochs"])
    epoch_loop = tqdm(loop, desc="Epochs", position=0, leave=True)

    for step in epoch_loop:
        model.train()
        running_train_acc = 0.0
        for i, (input, output) in enumerate(train_loader):
            if config["is_test_batch"] and i > 1:
                break

            input, output = input.to(device), output.to(device)

            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, output)
            loss.backward()
            optimizer.step()

            running_train_acc += mean_pixel_distance(outputs, output).mean().item()

            inner_progress = f"{i+1}/{len(train_loader)}"
            epoch_loop.set_description(f"Epochs (Batch: {inner_progress})")
            epoch_loop.refresh()

        train_acc = running_train_acc / 2 if config["is_test_batch"] else running_train_acc / len(train_loader)

        model.eval()
        running_val_acc = 0.0
        with torch.no_grad():

            # log_images(model, config, test_loader, device, step + 1, epoch_confidence_images)

            for i, (input, output) in enumerate(val_loader):
                if config["is_test_batch"] and i > 1:
                    break

                input, output = input.to(device), output.to(device)
                outputs = model(input)
                loss = criterion(outputs, output)

                running_val_acc += mean_pixel_distance(outputs, output).mean().item()

        val_acc = running_val_acc / 2 if config["is_test_batch"] else running_val_acc / len(val_loader)

        # if step + 1 in examples_tracking_epochs:
        #     with torch.no_grad():
        #         if track_image_count > 0:
        #             table_binary, table_continuous = create_examples_tables()
        #             log_examples_tables(
        #                 table_binary, table_continuous, model, test_loader, device, step + 1, track_image_count
        #             )
        #             save_examples_tables(
        #                 table_binary, table_continuous,
        #                 f"Training First N Examples/Binary Table Epoch {str(step + 1).zfill(3)}",
        #                 f"Training First N Examples/Continuous Table Epoch {str(step + 1).zfill(3)}"
        #             )

        #         if len(selected_tracking_images) > 0:
        #             table_binary, table_continuous = create_examples_tables()
        #             log_examples_tables(
        #                 table_binary, table_continuous, model, test_loader, device, step + 1,
        #                 image_names=selected_tracking_images
        #             )
        #             save_examples_tables(
        #                 table_binary,
        #                 table_continuous,
        #                 f"Training Selected Examples/Binary Table Epoch {str(step + 1).zfill(3)}",
        #                 f"Training Selected Examples/Continuous Table Epoch {str(step + 1).zfill(3)}"
        #             )

        wandb.log({
            "epoch": step + 1,
            "Acc/a_train_acc": train_acc,
            "Acc/b_val_acc": val_acc,
        })

        print(
            f"Epoch {step + 1}/{config['epochs']}, \
    Acc: {round(train_acc, 5)}, Validation Acc: {round(val_acc, 5)}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        else:
            num_epochs_without_improvement += 1

        if num_epochs_without_improvement >= config['patience']:
            print(f'Early stopping after {step + 1} epochs!')
            break

    epoch_loop.close()

    wandb.log({
        "Acc/c_best_val_acc": best_val_acc,
    })

    print(f"FINISHED! Best epoch: {best_epoch}, Best Accuracy: {best_val_acc}")


def save_model(model, model_name: str, model_time: str, path='./models/'):
    model.eval()
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(model, f'{path}model_{model_name}_{model_time}.pth')
    print('Model saved!')

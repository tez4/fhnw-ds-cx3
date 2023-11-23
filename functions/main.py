import wandb
import torch
import torch.nn as nn
from models import BaseNet, UNet
from datetime import datetime
from training import train_model, set_seed, save_model
from dataloader import get_data_loaders

if __name__ == "__main__":
    wandb.login()
    set_seed(42)

    config = {
        "name": "BaseNet",
        "epochs": 2,
        "image_size": 512,
        "train_batch_size": 8,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "dataset": "experiment_95_preprocessed",
        "lr": 1e-4,
        "is_test_batch": False,
        "start_time": datetime.now().strftime("%d.%m.%Y_%H%M"),
        "optimizer": 'SGD',
        "random_horizontal_flip": True,
        "num_workers": 0,
        "loss": "CE",
        "patience": 15,
    }

    wandb_config = {
        'entity': 'tez4',
        "project": "cx3",
        "group": "first_test",
        "tags": ["first_test"],
    }

    if config["loss"] == "CE":
        loss_func = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    model = BaseNet(3, 1)
    # model = torch.load("./models/all/model_small CNN_13.04.2023_0946.pth", map_location=torch.device('cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_loader, val_loader, test_loader = get_data_loaders(config=config, shuffle=True)

    train_model(
        model=model,
        optimizer=optimizer,
        criterion=loss_func,
        config=config,
        wandb_config=wandb_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    save_model(model, config["name"], config["start_time"])

    wandb.finish()

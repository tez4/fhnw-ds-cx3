import json
import wandb
import torch
import torch.nn as nn
from modelling import BaseNet, UNet, Discriminator
from pathlib import Path
from datetime import datetime
from training import set_seed, Trainer
from dataloader import get_data_loaders

if __name__ == "__main__":

    assert Path("./config_local.json").exists(), "config not found. copy config.json to create config_local.json!"
    with open("./config_local.json") as f:
        config = json.load(f)

    config["start_time"] = datetime.now().strftime("%d.%m.%Y_%H%M")

    wandb.login(key=config["wandb_api_key"])
    set_seed(42)

    if config["loss"] == "MSE":
        loss_func = nn.MSELoss()
    if config["loss"] == "L1":
        loss_func = nn.L1Loss()
    else:
        raise NotImplementedError()

    if config["multi_task_learning"]:
        if config["name"] == "BaseNet":
            model = BaseNet(3, 8)
        elif config["name"] == "UNet":
            model = UNet(3, 8, config['bilinear'], config['attention'])
        else:
            raise NotImplementedError()
    else:
        if config["name"] == "BaseNet":
            model = BaseNet(3, 3)
        elif config["name"] == "UNet":
            model = UNet(3, 3, config['bilinear'], config['attention'])
        else:
            raise NotImplementedError()

    if config["multi_task_learning"]:
        discriminator = Discriminator(11)
    else:
        discriminator = Discriminator(6)

    # model = torch.load("./models/model_UNet_24.11.2023_0033.pth", map_location=torch.device('cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    optimizer_discriminator = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_loader, val_loader, test_loader, real_loader = get_data_loaders(config=config, shuffle=True)

    model_trainer = Trainer(
        config, model, discriminator, optimizer, optimizer_discriminator, loss_func, loss_func,
        train_loader, val_loader, real_loader
    )
    for step in range(config["epochs"]):
        model_trainer.train(step)
        model_trainer.validate(step)
        model_trainer.log_values(step)

        if model_trainer.num_epochs_without_improvement >= config['patience']:
            print(f'Early stopping after {step + 1} epochs!')
            break

    model_trainer.finish_training(step)
    model_trainer.save_model()

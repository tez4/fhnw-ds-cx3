import wandb
import torch
import torch.nn as nn
from models import BaseNet, UNet
from datetime import datetime
from training import set_seed, Trainer
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
        "dataset": "experiment_93_preprocessed",
        "lr": 1e-4,
        "is_test_batch": False,
        "start_time": datetime.now().strftime("%d.%m.%Y_%H%M"),
        "optimizer": 'Adam',
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
        "table_images": [86, 87],
    }

    if config["loss"] == "CE":
        loss_func = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    model = BaseNet(3, 1)
    # model = torch.load("./models/all/model_small CNN_13.04.2023_0946.pth", map_location=torch.device('cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_loader, val_loader, test_loader = get_data_loaders(config=config, shuffle=True)

    model_trainer = Trainer(config, wandb_config, model, optimizer, loss_func, train_loader, val_loader)
    for step in range(config["epochs"]):
        model_trainer.train(step)
        model_trainer.validate(step)
        model_trainer.log_values(step)

        if model_trainer.num_epochs_without_improvement >= config['patience']:
            print(f'Early stopping after {step + 1} epochs!')
            break

    model_trainer.finish_training(step)
    model_trainer.save_model()

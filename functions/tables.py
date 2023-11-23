
import os
import cv2
import torch
import wandb
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from pathlib import Path
from scores import mean_pixel_distance


def create_examples_tables(model, data_loader, device, epoch, image_names, table_name):
    table = wandb.Table(
        columns=[
            "Name",
            "Input",
            "True Depth",
            "Predicted Depth",
            "Accuracy (distance)",
            "Epoch"
        ]
    )

    for (inputs, targets, image_path) in data_loader:

        if len([name for name in image_path if name.endswith(tuple(map(str, image_names)))]) == 0:
            continue

        inputs = inputs.to(device)
        outputs = model(inputs)

        inputs = inputs.to('cpu')
        outputs = outputs.to('cpu')

        pixel_acc = mean_pixel_distance(outputs, targets)

        array_inputs = np.array(inputs * 255)
        array_targets = np.array(targets * 255)
        array_outputs = np.array(outputs.detach() * 255)

        for i in range(array_inputs.shape[0]):

            if not image_path[i].endswith(tuple(map(str, image_names))):
                continue

            input_image = Image.fromarray(np.transpose(array_inputs[i], (1, 2, 0)).astype('uint8'), mode='RGB')
            depth_true = Image.fromarray(array_targets[i][0].astype("uint8"), mode='L')
            depth_pred = Image.fromarray(array_outputs[i][0].astype("uint8"), mode='L')

            table.add_data(
                image_path[i].split('/')[-1],
                wandb.Image(input_image),
                wandb.Image(depth_true),
                wandb.Image(depth_pred),
                round(pixel_acc[i].item(), 5),
                epoch
            )

    wandb.log({
        table_name: table,
    })
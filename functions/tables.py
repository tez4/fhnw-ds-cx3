
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
            "Pred Depth",
            "True Mask",
            "Pred Mask",
            "True Normals",
            "Pred Normals",
            "True Output",
            "Pred Output",
            "Accuracy",
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
        array_outputs = np.clip(np.array(outputs.detach() * 255), 0, 255)

        for i in range(array_inputs.shape[0]):

            if image_path[i].endswith(tuple(map(str, image_names))) is False:
                continue

            input_image = Image.fromarray(np.transpose(array_inputs[i], (1, 2, 0)).astype('uint8'), mode='RGB')
            depth_true = Image.fromarray(array_targets[i][0].astype("uint8"), mode='L')
            depth_pred = Image.fromarray(array_outputs[i][0].astype("uint8"), mode='L')
            mask_true = Image.fromarray(array_targets[i][1].astype("uint8"), mode='L')
            mask_pred = Image.fromarray(array_outputs[i][1].astype("uint8"), mode='L')
            normals_true = Image.fromarray(np.transpose(array_targets[i][2:5], (1, 2, 0)).astype("uint8"), mode='RGB')
            normals_pred = Image.fromarray(np.transpose(array_outputs[i][2:5], (1, 2, 0)).astype("uint8"), mode='RGB')
            p_1_true = Image.fromarray(np.transpose(array_targets[i][5:], (1, 2, 0)).astype("uint8"), mode='RGB')
            p_1_pred = Image.fromarray(np.transpose(array_outputs[i][5:], (1, 2, 0)).astype("uint8"), mode='RGB')

            table.add_data(
                image_path[i].split('/')[-1],
                wandb.Image(input_image),
                wandb.Image(depth_true),
                wandb.Image(depth_pred),
                wandb.Image(mask_true),
                wandb.Image(mask_pred),
                wandb.Image(normals_true),
                wandb.Image(normals_pred),
                wandb.Image(p_1_true),
                wandb.Image(p_1_pred),
                round(pixel_acc[i].item(), 5),
                epoch
            )

    wandb.log({
        table_name: table,
    })


def get_video_arrays(video_arrays, model, data_loader, device, image_names):
    if video_arrays is None:
        video_arrays = {}

    for (inputs, _, image_path) in data_loader:

        if len([name for name in image_path if name.endswith(tuple(map(str, image_names)))]) == 0:
            continue

        inputs = inputs.to(device)
        outputs = model(inputs)

        inputs = inputs.to('cpu')
        outputs = outputs.to('cpu')

        array_outputs = np.clip(np.array(outputs.detach() * 255), 0, 255)

        for i in range(array_outputs.shape[0]):

            if image_path[i].endswith(tuple(map(str, image_names))) is False:
                continue

            image_name = image_path[i].split('/')[-1]
            if image_name not in video_arrays:
                video_arrays[image_name] = {}

            depth = np.repeat(array_outputs[i][0].astype("uint8")[np.newaxis, ...], 3, axis=0)[np.newaxis, ...]
            mask = np.repeat(array_outputs[i][1].astype("uint8")[np.newaxis, ...], 3, axis=0)[np.newaxis, ...]
            normals = array_outputs[i][2:5].astype("uint8")[np.newaxis, ...]
            p_1 = array_outputs[i][5:].astype("uint8")[np.newaxis, ...]

            for array, name in zip([depth, mask, normals, p_1], ['depth', 'mask', 'normals', 'p_1']):
                if name not in video_arrays[image_name]:
                    video_arrays[image_name][name] = array
                else:
                    video_arrays[image_name][name] = np.concatenate((video_arrays[image_name][name], array), axis=0)

    return video_arrays


def create_video_tables(video_arrays, table_name):
    table = wandb.Table(
        columns=[
            "Name",
            "Depth",
            "Mask",
            "Normals",
            "Output",
        ]
    )

    for image_name in video_arrays:
        table.add_data(
            str(image_name),
            wandb.Video(video_arrays[image_name]['depth'], fps=24, format="mp4"),
            wandb.Video(video_arrays[image_name]['mask'], fps=24, format="mp4"),
            wandb.Video(video_arrays[image_name]['normals'], fps=24, format="mp4"),
            wandb.Video(video_arrays[image_name]['p_1'], fps=24, format="mp4"),
        )

    wandb.log({
        table_name: table,
    })

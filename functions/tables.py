
import os
import cv2
import torch
import wandb
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from pathlib import Path
from scores import mean_squared_error


def create_real_tables(model, data_loader, device, epoch, table_name, multi_task_learning):
    if multi_task_learning:
        table_columns = [
            "Name",
            "Input",
            "Pred Depth",
            "Pred Mask",
            "Pred Normals",
            "Pred Output",
            "Epoch"
        ]
    else:
        table_columns = [
            "Name",
            "Input",
            "Pred Output",
            "Epoch"
        ]

    table = wandb.Table(columns=table_columns)

    for (inputs, image_path) in data_loader:

        inputs = inputs.to(device)
        outputs = model(inputs)

        inputs = inputs.to('cpu')
        outputs = outputs.to('cpu')

        array_inputs = np.array(inputs * 255)
        array_outputs = np.clip(np.array(outputs.detach() * 255), 0, 255)

        for i in range(array_inputs.shape[0]):

            input_image = Image.fromarray(np.transpose(array_inputs[i], (1, 2, 0)).astype('uint8'), mode='RGB')

            if multi_task_learning:
                depth_pred = Image.fromarray(array_outputs[i][0].astype("uint8"), mode='L')
                mask_pred = Image.fromarray(array_outputs[i][1].astype("uint8"), mode='L')
                normals_pred = Image.fromarray(
                    np.transpose(array_outputs[i][2:5], (1, 2, 0)).astype("uint8"), mode='RGB'
                )

            p_1_pred = Image.fromarray(np.transpose(array_outputs[i][-3:], (1, 2, 0)).astype("uint8"), mode='RGB')

            if multi_task_learning:
                table.add_data(
                    image_path[i].split('/')[-1],
                    wandb.Image(input_image),
                    wandb.Image(depth_pred),
                    wandb.Image(mask_pred),
                    wandb.Image(normals_pred),
                    wandb.Image(p_1_pred),
                    epoch
                )
            else:
                table.add_data(
                    image_path[i].split('/')[-1],
                    wandb.Image(input_image),
                    wandb.Image(p_1_pred),
                    epoch
                )

    wandb.log({
        table_name: table,
    })


def create_examples_tables(model, data_loader, device, epoch, table_name, multi_task_learning, n):
    if multi_task_learning:
        table_columns = [
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
    else:
        table_columns = [
            "Name",
            "Input",
            "True Output",
            "Pred Output",
            "Accuracy",
            "Epoch"
        ]

    table = wandb.Table(columns=table_columns)

    example_image_paths = data_loader.dataset.image_paths[:n]

    for (inputs, targets, image_path) in data_loader:

        if len([name for name in image_path if name in example_image_paths]) == 0:
            continue

        inputs = inputs.to(device)
        outputs = model(inputs)

        inputs = inputs.to('cpu')
        outputs = outputs.to('cpu')

        pixel_acc = mean_squared_error(outputs, targets)

        array_inputs = np.array(inputs * 255)
        array_targets = np.array(targets * 255)
        array_outputs = np.clip(np.array(outputs.detach() * 255), 0, 255)

        for i in range(array_inputs.shape[0]):

            if image_path[i] not in example_image_paths:
                continue

            input_image = Image.fromarray(np.transpose(array_inputs[i], (1, 2, 0)).astype('uint8'), mode='RGB')

            if multi_task_learning:
                depth_true = Image.fromarray(array_targets[i][0].astype("uint8"), mode='L')
                depth_pred = Image.fromarray(array_outputs[i][0].astype("uint8"), mode='L')
                mask_true = Image.fromarray(array_targets[i][1].astype("uint8"), mode='L')
                mask_pred = Image.fromarray(array_outputs[i][1].astype("uint8"), mode='L')
                normals_true = Image.fromarray(
                    np.transpose(array_targets[i][2:5], (1, 2, 0)).astype("uint8"), mode='RGB'
                )
                normals_pred = Image.fromarray(
                    np.transpose(array_outputs[i][2:5], (1, 2, 0)).astype("uint8"), mode='RGB'
                )

            p_1_true = Image.fromarray(np.transpose(array_targets[i][-3:], (1, 2, 0)).astype("uint8"), mode='RGB')
            p_1_pred = Image.fromarray(np.transpose(array_outputs[i][-3:], (1, 2, 0)).astype("uint8"), mode='RGB')

            if multi_task_learning:
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
            else:
                table.add_data(
                    image_path[i].split('/')[-1],
                    wandb.Image(input_image),
                    wandb.Image(p_1_true),
                    wandb.Image(p_1_pred),
                    round(pixel_acc[i].item(), 5),
                    epoch
                )

    wandb.log({
        table_name: table,
    })


def get_video_arrays(video_arrays, model, data_loader, device, multi_task_learning, n):
    if video_arrays is None:
        video_arrays = {}

    example_image_paths = data_loader.dataset.image_paths[:n]

    for (inputs, _, image_path) in data_loader:

        if len([name for name in image_path if name in example_image_paths]) == 0:
            continue

        inputs = inputs.to(device)
        outputs = model(inputs)

        inputs = inputs.to('cpu')
        outputs = outputs.to('cpu')

        array_outputs = np.clip(np.array(outputs.detach() * 255), 0, 255)

        for i in range(array_outputs.shape[0]):

            if image_path[i] not in example_image_paths:
                continue

            image_name = image_path[i].split('/')[-1]
            if image_name not in video_arrays:
                video_arrays[image_name] = {}

            if multi_task_learning:
                depth = np.repeat(array_outputs[i][0].astype("uint8")[np.newaxis, ...], 3, axis=0)[np.newaxis, ...]
                mask = np.repeat(array_outputs[i][1].astype("uint8")[np.newaxis, ...], 3, axis=0)[np.newaxis, ...]
                normals = array_outputs[i][2:5].astype("uint8")[np.newaxis, ...]

            p_1 = array_outputs[i][-3:].astype("uint8")[np.newaxis, ...]

            if multi_task_learning:
                arrays = [depth, mask, normals, p_1]
                names = ['depth', 'mask', 'normals', 'p_1']
            else:
                arrays = [p_1]
                names = ['p_1']

            for array, name in zip(arrays, names):
                if name not in video_arrays[image_name]:
                    video_arrays[image_name][name] = array
                else:
                    video_arrays[image_name][name] = np.concatenate((video_arrays[image_name][name], array), axis=0)

    return video_arrays


def create_video_tables(video_arrays, table_name, multi_task_learning):
    if multi_task_learning:
        table_columns = [
            "Name",
            "Depth",
            "Mask",
            "Normals",
            "Output"
        ]
    else:
        table_columns = [
            "Name",
            "Output"
        ]

    table = wandb.Table(columns=table_columns)

    for image_name in video_arrays:
        if multi_task_learning:
            table.add_data(
                str(image_name),
                wandb.Video(video_arrays[image_name]['depth'], fps=24, format="mp4"),
                wandb.Video(video_arrays[image_name]['mask'], fps=24, format="mp4"),
                wandb.Video(video_arrays[image_name]['normals'], fps=24, format="mp4"),
                wandb.Video(video_arrays[image_name]['p_1'], fps=24, format="mp4"),
            )
        else:
            table.add_data(
                str(image_name),
                wandb.Video(video_arrays[image_name]['p_1'], fps=24, format="mp4"),
            )

    wandb.log({
        table_name: table,
    })

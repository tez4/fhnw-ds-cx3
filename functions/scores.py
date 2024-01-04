import io
import torch
import numpy as np
import scipy.ndimage as ndi


def mean_squared_error(outputs, targets):
    assert outputs.shape == targets.shape, "output and target shapes must be equal"
    assert len(outputs.shape) == 4, "outputs and targets must be 4D tensors (batch, channels, height, width)"

    differences = (outputs - targets) ** 2
    mse_per_image = differences.mean(dim=[1, 2, 3])

    return mse_per_image

from __future__ import absolute_import

import os
import shutil
import torch
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def save_checkpoint(state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar',
                    snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    # if snapshot and snapshot!=0 and state['epoch'] % snapshot == 0:
    if snapshot:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
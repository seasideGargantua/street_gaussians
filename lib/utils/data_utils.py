import numpy as np
import torch
import os
import math
from lib.utils.graphics_utils import focal2fov
from lib.datasets.base_readers import CameraInfo
from PIL import Image
from tqdm import tqdm

def to_cuda(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b) for b in batch]
        return batch
    elif isinstance(batch, torch.Tensor):
        return batch.cuda()
    elif isinstance(batch, np.ndarray):
        return torch.from_numpy(batch).cuda()
    elif isinstance(batch, dict):
        for k in batch:
            if k == 'meta':
                continue
            batch[k] = to_cuda(batch[k])
        return batch
    else:
        raise NotImplementedError

def get_split_data(split_train, split_test, data):
    if split_train != -1:
        train_data = [d for idx, d in enumerate(data) if idx % split_train == 0]
        test_data = [d for idx, d in enumerate(data) if idx % split_train != 0]
    else:
        train_data = [d for idx, d in enumerate(data) if idx % split_test != 0]
        test_data = [d for idx, d in enumerate(data) if idx % split_test == 0]
    return train_data, test_data

def get_val_frames(num_frames: int, test_every: int, train_every: int):
    if train_every is None or train_every < 0:
        val_frames = set(np.arange(test_every, num_frames, test_every))
        train_frames = (set(np.arange(num_frames)) - val_frames) if test_every > 1 else set()
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else set()

    train_frames = sorted(list(train_frames))
    val_frames = sorted(list(val_frames))

    return train_frames, val_frames

def get_rays(H, W, K, R, T, perturb=False):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    
    if perturb:
        perturb_i = np.random.rand(H, W)
        perturb_j = np.random.rand(H, W)
        xy1 = np.stack([i + perturb_i, j + perturb_j, np.ones_like(i)], axis=2)
    else:
        xy1 = np.stack([i + 0.5, j + 0.5, np.ones_like(i)], axis=2)
    
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def sphere_intersection(rays_o, rays_d, center, radius):
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    b = np.sum((rays_o - center) * rays_d, axis=-1, keepdims=True)
    c = np.sum((rays_o - center) * (rays_o - center), axis=-1, keepdims=True) - radius ** 2
    
    nears = (-b - np.sqrt(b ** 2 - c))
    fars = (-b + np.sqrt(b ** 2 - c))
    
    nears = np.nan_to_num(nears, nan=0.0)
    fars = np.nan_to_num(fars, nan=1e3)
    
    p_sphere = rays_o + fars * rays_d 
    
    return p_sphere


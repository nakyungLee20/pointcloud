"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

def collate_dim(batch):
    max_points = max(sample['coord'].shape[0] for sample in batch)
    
    def pad_array(array, max_points, key):
        pad_size = max_points - array.shape[0]
        array = array.numpy().astype(np.float32)
        if pad_size > 0:
            if key == "instance" or key == "segment":
                array = array.astype(np.int32)
                padding = np.full((pad_size,), -1)
                array = np.hstack((array, padding))
            else:
                padding = np.zeros((pad_size, array.shape[1]), dtype=array.dtype)
                array = np.vstack((array, padding))
        return array
    
    coords = []
    colors = []
    instances = []
    segments = []
    names = []
    
    for sample in batch:
        print(sample.keys())
        coords.append(pad_array(sample['coord'], max_points, "coord"))
        colors.append(pad_array(sample['color'], max_points, "color"))
        instances.append(pad_array(sample['instance'], max_points, "instance"))
        segments.append(pad_array(sample['segment'], max_points, "segment"))
        names.append(sample['name'])
    
    coords = torch.tensor(coords)
    colors = torch.tensor(colors)
    instances = torch.tensor(instances)
    segments = torch.tensor(segments)
        
    new_samples = {"coord": coords, "color": colors, "instance": instances, "segment": segments, "name": names}   
    
    return new_samples


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    # batch = collate_fn(batch)
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))

"""
Data loader for EGA-Ploc protein subcellular localization / EGA-Ploc蛋白质亚细胞定位的数据加载器
This module handles dataset construction and distributed data loading for training and evaluation.
此模块处理训练和评估的数据集构建和分布式数据加载。
"""

import numpy as np
import random

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import utils.distributed as du

from .build import build_dataset

from utils.config_defaults import get_cfg



def construct_loader(cfg, filePath, model_name, condition="normal", database="IHC", aug=False, shuffle=False, drop_last=False):
    assert condition in ["normal", "pathology"]
    dataset_name = cfg.DATA.DATASET

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, filePath, condition, database, aug)

    # Create a sampler for multi-process training
    world_size = du.get_world_size()
    sampler = DistributedSampler(dataset) if world_size > 1 else None

    # Create a loader
    if sampler is not None:
        loader = torch.utils.data.DataLoader(
            dataset,
            # batch_size=batch_size,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # Whether multiple processes read data（DataLoader default 0)
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,  # If True will place the data on the GPU（DataLoader default false）
            drop_last=drop_last,  # Whether to discard the last batch of data when the number of samples is not divisible by the batchsize（default: False)
            prefetch_factor=2,
            # worker_init_fn=seed_worker
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            # batch_size=batch_size,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # Whether multiple processes read data（DataLoader default 0)
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,  # If True will place the data on the GPU（DataLoader default false）
            drop_last=drop_last,
            # Whether to discard the last batch of data when the number of samples is not divisible by the batchsize（default: False)
            prefetch_factor=2,
        )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """
    Shuffles the dataset.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

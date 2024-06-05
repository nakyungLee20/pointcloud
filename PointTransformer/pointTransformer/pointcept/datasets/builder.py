"""
Dataset Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry
# from Customed.BHDataset import BHDataset

DATASETS = Registry("datasets")
# DATASETS.register('BHDataset', BHDataset)

def build_dataset(cfg):
    """Build datasets."""
    return DATASETS.build(cfg)

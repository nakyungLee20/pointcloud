import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections.abc import Sequence

from pointcept.datasets.builder import DATASETS, build_dataset
from pointcept.datasets.transform import Compose, TRANSFORMS

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch

import random
from collections.abc import Mapping

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )

# training
if __name__ == "__main__":
    main()
    # python tools/test.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base weight=exp/scannet/semseg-pt-v2m2-0-base/model/model_best.pth
    # python tools/train.py --config-file /home/leena/ptv3/Pointcept/Customed/semseg-pt-v3m1-bh.py --num-gpus 1 --options save_path=/home/leena/ptv3/Pointcept/Customed/seg_model resume=True weight=/home/leena/ptv3/Pointcept/Customed/model_best_nu.pth

# python trained.py --config-file /home/leena/ptv3/Pointcept/Customed/config_bh.py --num-gpus 1
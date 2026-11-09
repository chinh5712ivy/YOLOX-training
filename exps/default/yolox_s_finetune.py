#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-S fine-tuning experiment for fire-alert-v6a dataset.

Pretrained from: yolox_s_fire_smoke_v5.pth (Best model V1)
Key changes from base config:
  - num_classes: 2 (fire + smoke)
  - max_epoch: 40
  - basic_lr_per_img: 20% of original (0.002 / 64)
  - warmup_epochs: 1
  - no_aug_epochs: 8
  - eval_interval: 1 (evaluate every epoch to monitor overfitting)
  - Dataset: fire-alert-v6a with v6a annotation files
"""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 2  # fire, smoke
        self.depth = 0.33    # YOLOX-S architecture
        self.width = 0.50    # YOLOX-S architecture

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.multiscale_range = 0
        self.data_dir = r"D:\OJT_F_Software\ojt\Mapping\SDT\fire-data\dataset\fire-alert-v8"
        self.train_ann = "instances_train2017_v8.json"
        self.val_ann = "instances_val2017_v8.json"
        self.test_ann = "instances_test2017_v8.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.2
        self.mixup_prob = 0.0
        self.hsv_prob = 0.3
        self.flip_prob = 0.5
        self.degrees = 3.0
        self.translate = 0.05
        self.mosaic_scale = (0.8, 1.2)
        self.enable_mixup = False
        self.mixup_scale = (0.5, 1.5)
        self.shear = 0.0

        # -------------- training config ------------------- #
        self.max_epoch = 30
        # 20% of original LR: 0.01/64 * 0.2 = 0.002/64
        self.basic_lr_per_img = 0.002 / 64.0
        self.warmup_epochs = 1        # 1 epoch warmup for stability
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        self.scheduler = "yoloxwarmcos"  # Cosine annealing with warmup
        self.no_aug_epochs = 5         # Disable Mosaic/MixUp for last 8 epochs
        self.ema = True                # EMA for stable evaluation weights

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1          # Evaluate every epoch to track overfitting
        self.save_history_ckpt = True

        # -------------- early stopping config ------------- #
        self.early_stopping_patience = 8   # Stop fine-tuning if mAP does not improve for 8 consecutive epochs
        self.early_stopping_min_delta = 0.001  # Minimum improvement threshold (0.1% mAP)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

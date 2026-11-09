#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-S training experiment from base/scratch on fire-alert-v8 dataset.

Full official YOLOX-S hyperparameters:
  - num_classes: 2 (fire + smoke)
  - basic_lr_per_img: 0.01 / 64.0 (full original learning rate)
  - warmup_epochs: 5
  - max_epoch: 100
  - no_aug_epochs: 15 (disable mosaic/mixup for last 15 epochs)
  - Full data augmentations: Mosaic (1.0), MixUp (1.0), Multi-scale (range 5)
  - Dataset: fire-alert-v8
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
        self.multiscale_range = 5  # Enable multi-scale training [640 - 5*32, 640 + 5*32]
        self.data_dir = r"D:\OJT_F_Software\ojt\Mapping\SDT\fire-data\dataset\fire-alert-v8"
        self.train_ann = "instances_train2017_v8.json"
        self.val_ann = "instances_val2017_v8.json"
        self.test_ann = "instances_test2017_v8.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0       # Full mosaic augmentation
        self.mixup_prob = 1.0        # Full mixup augmentation
        self.hsv_prob = 1.0          # Full HSV color jittering
        self.flip_prob = 0.5         # Horizontal flip
        self.degrees = 10.0          # Rotation range (-10, 10)
        self.translate = 0.1         # Translation range (-0.1, 0.1)
        self.mosaic_scale = (0.1, 2) # Mosaic scale range
        self.enable_mixup = True     # Enable mixup
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0             # Shear range (-2, 2)

        # -------------- training config ------------------- #
        self.max_epoch = 100         # 100 epochs for full training
        self.basic_lr_per_img = 0.01 / 64.0  # Official full YOLOX learning rate
        self.warmup_epochs = 5       # 5 epochs warmup for stability
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        self.scheduler = "yoloxwarmcos"  # Cosine annealing with warmup
        self.no_aug_epochs = 15      # Close mosaic/mixup for last 15 epochs to stabilize convergence
        self.ema = True              # Exponential Moving Average for evaluation weights

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1       # Evaluate every epoch to track mAP
        self.save_history_ckpt = True

        # -------------- early stopping config ------------- #
        self.early_stopping_patience = 15  # Stop training if mAP does not improve for 15 consecutive epochs
        self.early_stopping_min_delta = 0.001  # Minimum improvement threshold (0.1% mAP)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65


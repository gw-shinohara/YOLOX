#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/home/shinohara/Documents/YOLOX/datasets/white_cane_detection"
        self.train_ann = "grasped_dataset_train_no_obj.json"
        self.val_ann = "grasped_sub_dataset_val.json"

        self.num_classes = 13

        self.max_epoch = 30

        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 999

        self.data_num_workers = 4
        self.eval_interval = 1

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset

        return COCODataset(
            data_dir="/home/shinohara/Downloads/grasped_dataset_train_no_obj",
            json_file=self.train_ann,
            name="aug",
            img_size=self.input_size,
            preproc=None,
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir="/home/shinohara/Documents/YOLOX/datasets/white_cane_detection",
            json_file=self.val_ann if not testdev else self.test_ann,
            name="2nd" if not testdev else "2nd",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model

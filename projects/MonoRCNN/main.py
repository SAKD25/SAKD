#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import copy
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
import numpy as np

from lib.train_test import (
    add_monodet_config, 
    register_dataset, 
    MonoDetMapper,
    inference_on_dataset,
    evaluate_on_dataset,
    set_random_seed,
)

from lib.all_roi_heads import ALLROIHeads
from lib.all_roi_heads_harmonize import ALLROIHeads_harmonize
from lib.all_roi_heads_teacher import ALLROIHeads_teacher
from lib.all_roi_heads_SAKD import ALLROIHeads_SAKD

from lib.att_head import ATTHead
from lib.dis_head import DISHead
from detectron2.modeling.meta_arch import Embed, ConvReg, LinearEmbed, SelfA

import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger("detectron2")
monodet_mapper_train = None
monodet_mapper_test = None

def do_test(cfg, model, iteration=None):
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=monodet_mapper_test)
        result_dir, iteration = inference_on_dataset(cfg, model, data_loader, iteration)
        comm.synchronize()
        #if comm.is_main_process():
        if dataset_name == 'KITTI_val1_val' :
            evaluate_on_dataset(cfg, result_dir, iteration)
            logger.info("Evaluation for {} is completed.".format(dataset_name))
        elif dataset_name == 'KITTI_demo_val':
            vis_dir = os.path.join(cfg.OUTPUT_DIR, 'evaluation', iteration, 'visualization')
            print("visualization save at:",vis_dir)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_monodet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    cfg = setup(args)
    set_random_seed(cfg.SEED)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'model')):
        os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'model'))
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'evaluation')):
        os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'evaluation'))
    register_dataset(cfg)
    global monodet_mapper_train, monodet_mapper_test
    monodet_mapper_train = MonoDetMapper(cfg, is_train=True)
    monodet_mapper_test = MonoDetMapper(cfg, is_train=False)
    model = build_model(cfg)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)


    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=os.path.join(cfg.OUTPUT_DIR, 'model')).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        do_test(cfg, model)
        return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

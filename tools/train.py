"""
Training script for EGA-Ploc protein subcellular localization / EGA-Ploc蛋白质亚细胞定位的训练脚本
This module handles distributed training of EGA-Ploc models for multi-label classification.
此模块处理EGA-Ploc模型的多标签分类分布式训练。
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import random

import numpy as np
import torch
import torch.nn as nn
from timm.models import create_model
import torch.distributed as dist

import utils.distributed as du
import utils.checkpoint as cu
from datasets.loader import construct_loader
from models.classifier_model import getClassifier
from models.losses import get_loss_func
from models.train_classifier import (load_best_classifier_model, train, train_DeePSLoc, warmup,
                                     load_best_pretrain_classifier_model)
from utils.args import parse_args
from utils.config_defaults import get_cfg, dir_prefixs
from utils.optimizer import construct_optimizer, get_optimizer_func
from utils.scheduler import construct_scheduler, get_scheduler_func


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = get_cfg()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        torch.cuda.set_device(torch.distributed.get_rank())
        device=torch.device("cuda", torch.distributed.get_rank())
    world_size = du.get_world_size()

    if du.is_master_proc():
        print('use {} gpus!'.format(world_size))

    random.seed(cfg.RNG_SEED + args.local_rank)
    np.random.seed(cfg.RNG_SEED + args.local_rank)
    torch.manual_seed(cfg.RNG_SEED + args.local_rank)
    torch.cuda.manual_seed(cfg.RNG_SEED + args.local_rank)
    torch.cuda.manual_seed_all(cfg.RNG_SEED + args.local_rank)

    for classifier_model in cfg.TRAIN.CLASSIFIER_NAME:
        for database in cfg.DATA.DATASET_NAME:
            multilabel = True

            path_prefix = dir_prefixs[database]
            result_prefix = "{}/results/{}".format(cfg.DATA.RESULT_DIR, database)
            log_prefix = "{}".format(database)
            if du.is_master_proc():
                print(log_prefix)

            train_file_path = "%s_train.csv" % path_prefix
            val_file_path = "%s_test.csv" % path_prefix


            """ Constructing classifier model """
            # Classifier
            model = getClassifier(cfg, model_name=classifier_model, pretrain=False)
            model = model.to(device)
            if world_size > 1:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())

            """ load data """
            train_loader = construct_loader(cfg, train_file_path, model_name=classifier_model, condition="normal", database=database,
                                            aug=True, shuffle=True, drop_last=False)
            val_loader = construct_loader(cfg, val_file_path, model_name=classifier_model, condition="normal", database=database,
                                          shuffle=False, drop_last=False)

            """ Training the classifier model """
            if cfg.CLASSIFIER.TRAIN:
                if du.is_master_proc():
                    print("Train classifier {}".format(classifier_model))


                """ Constructing optimizier """
                if "wd-005" in classifier_model:
                    optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.05, amsgrad=False)
                elif "wd-001" in classifier_model:
                    optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.01, amsgrad=False)
                elif "wd-0005" in classifier_model:
                    optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.005, amsgrad=False)
                else:
                    optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0, amsgrad=False)

                """ Constructing scheduler """
                scheduler = construct_scheduler(cfg.CLASSIFIER, optimizer, "warmupCosine")
                scaler = torch.cuda.amp.GradScaler(enabled=True)


                """ Loss function """
                nums = None
                totalNums = 0

                if "mlce" in classifier_model:
                    criterion = get_loss_func("multilabel_balanced_cross_entropy")(reduction="none", nums=nums, total_nums=totalNums).to(device)
                elif "bce" in classifier_model:
                    criterion = get_loss_func("bce_logit")(reduction="none").to(device)
                else:
                    criterion = get_loss_func("multilabel_categorical_cross_entropy")(reduction="none", nums=nums, total_nums=totalNums).to(device)

                start_epoch = 0
                min_loss = float("inf")
                if cfg.CLASSIFIER.CKP:
                    checkpoint_path = "{}/{}/latest_model.pth".format(result_prefix, classifier_model)
                    start_epoch, min_loss = cu.load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)

                dist.barrier()

                train(cfg, None, model, loader=[train_loader, val_loader],
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler, criterion=criterion, l1_alpha=cfg.CLASSIFIER.L1_ALPHA, l2_alpha=cfg.CLASSIFIER.L2_ALPHA,
                    epoch=cfg.CLASSIFIER.EPOCH_NUM, start_epoch=start_epoch, min_loss=min_loss, model_name=classifier_model, patch_size=cfg.SAE.MODEL_NAME, multilabel=multilabel,
                    result_prefix=result_prefix, log_prefix=log_prefix, train_file_path=train_file_path, val_file_path=val_file_path,
                    device=device
                )


if __name__ == "__main__":
    main()

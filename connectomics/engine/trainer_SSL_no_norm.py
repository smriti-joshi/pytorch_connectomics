from __future__ import print_function, division
from typing import Optional

import os
import time
import GPUtil
import numpy as np
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .solver import *
from ..model import *
from ..utils.monitor import build_monitor
from ..data.augmentation import build_train_augmentor, TestAugmentor, build_uda_augmentor
from ..data.dataset import build_dataloader, get_dataset
from ..data.dataset.build import build_dataloader_uda
from ..data.utils import build_blending_matrix, writeh5
from ..data.utils import get_padsize, array_unpad
from ..engine.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

class TrainerUDA_2(Trainer):
    r"""Trainer class for supervised learning.
    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """
    def __init__(self, 
                 cfg: CfgNode, 
                 device: torch.device, 
                 mode: str = 'train', 
                 rank: Optional[int] = None, 
                 checkpoint: Optional[str] = None):

        super().__init__(cfg, device, mode, rank,
                      checkpoint)

        # self.lmd = 4
        assert mode in ['train', 'test']
        
        self.augmentor_ssl = build_uda_augmentor(self.cfg)
        self.criterion_uda = Criterion.build_from_cfg_uda(self.cfg, self.device)

        self.dataset, self.dataloader_ssl = None, None
        
        if cfg.DATASET.DO_CHUNK_TITLE == 0:
            self.dataloader_ssl = build_dataloader_uda(self.cfg, self.augmentor_ssl, self.mode, rank=rank)
            self.dataloader_ssl = iter(self.dataloader_ssl)
            if self.mode == 'train' and cfg.DATASET.VAL_IMAGE_NAME is not None:
                self.val_loader = build_dataloader(self.cfg, None, mode='val', rank=rank)

    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()
        sigmoid = nn.Sigmoid()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            ##### Supervised #####

            # load data
            sample = next(self.dataloader)
            volume = sample.out_input
            target, weight = sample.out_target_l, sample.out_weight_l
            self.data_time = time.perf_counter() - self.start_time

            # prediction
            volume = volume.to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                pred = self.model(volume)
                loss_sup, losses_vis = self.criterion(pred, target, weight)
            
            ##### Unsupervised #####

            # load data
            sample_uda = next(self.dataloader_ssl)
            aug_volume_one, aug_volume_two, volume_track, volume_aug_track = sample_uda.out_input_one, sample_uda.out_input_two, sample_uda.out_volume_track, sample_uda.out_volume_aug_track

            # prediction
            aug_volume_one = aug_volume_one.to(self.device, non_blocking=True)
            aug_volume_two = aug_volume_two.to(self.device, non_blocking=True)
            volume_track = volume_track.to(self.device, non_blocking=True)
            volume_aug_track = volume_aug_track.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                pred_one = self.model(aug_volume_one)
                pred_two = self.model(aug_volume_two)
                pred_track = self.model(volume_track)
                pred_aug_track = self.model(volume_aug_track)
                # loss_ssl = float(self.weighted_mse_loss(pred_one * mask, pred_two * mask))

                loss_entrp = self.entropy_loss(pred, pred_one, pred_two)
               
                loss_ssl = self.weighted_mse_loss(pred_two, pred_one, 0.8)


            # loss = loss_sup +  10 * loss_ssl  #+ 0.0001 * loss_entrp log2021-04-24_17-55-30
            loss = loss_sup + 10 * loss_ssl + + 0.0001 * loss_entrp

            losses_vis['uda_mse_loss'] = loss_ssl
            losses_vis['sup_loss'] = loss_sup
            losses_vis['entrp_loss'] = loss_entrp

            self._train_misc(loss, pred, volume, target, weight,iter_total, losses_vis, aug_volume_one,aug_volume_two, 
                pred_one, pred_two, volume_track, pred_track, volume_aug_track, pred_aug_track)
            
            if i % 10 == 0:
                print('[Iteration %05d] supervised_loss=%.5f' % (i, loss_sup))
        self.maybe_save_swa_model()

    def weighted_mse_loss(self, pred, target, thresh, weight=None):
        pred = torch.sigmoid(pred)
        target = torch.sigmoid(target)

        mask = (pred > thresh) * (target > thresh)

        pred = pred * mask
        target = target * mask

        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).to(pred.device)
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        return torch.sum(weight * (pred - target) ** 2) / norm_term

    def entropy_loss(self, pred, pred_one, pred_two):

        pred = torch.sigmoid(pred)
        pred_one = torch.sigmoid(pred_one)
        pred_two = torch.sigmoid(pred_two)
        
        entropy = torch.abs(torch.sum(-pred * torch.log(pred +(pred == 0).float())) 
                            - 0.5 * (torch.sum(-pred_one * torch.log(pred_one + (pred_one == 0).float())) + 
                            torch.sum(-pred_two * torch.log(pred_two + (pred_two == 0).float()))))
        return entropy
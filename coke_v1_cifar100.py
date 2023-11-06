# -*- coding: utf-8 -*-
"""COKE_v1_CIFAR100.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1thmxh_cqMMau2iuu5EWTmRCd_tdMHXmm
"""

#from google.colab import drive
#drive.mount('/content/drive')

#:wq
#cd /content/drive/MyDrive/CoKe/CoKe

import argparse
import builtins
import os
import random
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

#import coke.loader
#import coke.folder
#import coke.builder_single_view
import torch.nn.functional as F
#import coke.optimizer
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import easydict

from torch.utils.data import Dataset
from torchvision import datasets
import torch.distributed as dist

# 프로세스 그룹 초기화
dist.init_process_group(backend='gloo', init_method='tcp://localhost:12345', world_size=1, rank=0)

#from torch.utils.tensorboard import SummaryWriter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

args = easydict.EasyDict({
    'arch': 'resnet18',
    "workers" : 0, #현재 시스템 최대 병렬로 작업자 수
    "epochs" : 201,
    "start_epoch":0,
    "batch_size": 128,
    "lr": 0.2,
    "momentum": 0.9,
    "wd" : 1e-6,
    "weight_decay": 1e-6,
    "p" : 100,
    "print_freq": 100,
    "resume": '',
    "log": 'dim20_coke_single_view_cifar100',
    "coke_dim": 100,
    "coke_num_ins" : 50000,
    "coke_num_head": 10,
    # "coke_k": [10,20,30,40,50,60,70,80,90,100, 110,120,130,140, 150,160,170, 180,190,200],
    "coke_k": [20,40,60,80,100, 120, 140, 160, 180, 200],
    "coke_t": 0.1, #temperature
    "coke_dual_lr" : 0.1,
    "coke_stage": 202,
    "coke_tt" : 0.5,
    "coke_ratio": 0.9,
    "coke_ls": 5,
    "coke_beta": 0.5,
    "coke_alpha": 0.2})

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# modify from
# https://github.com/facebookresearch/moco-v3/blob/main/moco/loader.py

from PIL import Image, ImageFilter, ImageOps
import random


class SingleCropsTransform:
    """Take a single random crop of one image"""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x)

class DoubleCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class MultiCropsTransform:
    """Take multiple random crops of one image"""

    def __init__(self, base_transform1, base_transform2, small_transform, snum):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.small_transform = small_transform
        self.snum = snum

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        simgs = []
        for i in range(0, self.snum):
            simgs.append(self.small_transform(x))
        return [im1, im2, simgs]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoKe(nn.Module):
    """
    Build a CoKe model with multiple clustering heads
    """

    def __init__(self, base_encoder, K, dim=128, num_ins=50000, num_head=10, T=0.1, dual_lr=20, stage=100, t=0.5,
                 ratio=0.4, ls=5):
        super(CoKe, self).__init__()
        self.T = T
        self.K = K
        self.dual_lr = dual_lr
        self.ratio = ratio
        self.lb = [ratio / k for k in self.K]
        self.dual_lr = dual_lr
        self.ls = ls  # non-zero label size in second stage
        self.stage = stage  # number of epochs for the first stage
        self.t = t  # temperature for label smoothing
        self.num_head = num_head
        # create the encoder with projection head
        self.encoder = base_encoder(num_classes=dim)
        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim), nn.BatchNorm1d(dim))
        # prediction head
        self.predictor = nn.Sequential(nn.Linear(dim, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                       nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim))

        # decoupled cluster assignments
        self.pre_centers = []
        self.cur_centers = []
        self.duals = []
        self.counters = []
        for i in range(0, self.num_head):
            centers = F.normalize(torch.randn(dim, self.K[i]), dim=0)
            self.register_buffer("pre_center_" + str(i), centers.clone())
            self.register_buffer("cur_center_" + str(i), centers.clone())
            self.register_buffer("dual_" + str(i), torch.zeros(self.K[i]))
            self.register_buffer("counter_" + str(i), torch.zeros(self.K[i]))
        self.register_buffer("assign_labels", torch.ones(num_head, num_ins, ls, dtype=torch.long))
        self.register_buffer("label_val", torch.zeros(num_head, num_ins, ls))
        self.register_buffer("label_idx", torch.zeros(num_head, num_ins, dtype=torch.long))

    @torch.no_grad()
    def load_param(self):
        for i in range(0, self.num_head):
            self.pre_centers.append(getattr(self, "pre_center_" + str(i)))
            self.cur_centers.append(getattr(self, "cur_center_" + str(i)))
            self.duals.append(getattr(self, "dual_" + str(i)))
            self.counters.append(getattr(self, "counter_" + str(i)))

    @torch.no_grad()
    def gen_label(self, feats, epoch, branch):
        if epoch >= self.stage:
            return torch.argmax(torch.mm(feats, self.pre_centers[branch]) + self.duals[branch], dim=1).squeeze(-1)
        else:
            return torch.argmax(torch.mm(feats, self.cur_centers[branch]) + self.duals[branch], dim=1).squeeze(-1)

    @torch.no_grad()
    def update_label(self, targets, labels, epoch, branch):
        if epoch < self.stage or self.ls == 1:
            self.assign_labels[branch][targets, 0] = labels
        else:
            if epoch == self.stage:
                self.assign_labels[branch][targets, 0] = labels
                self.label_val[branch][targets, 0] = 1.
                self.label_idx[branch][targets] = 1
            else:
                factor = 1. / (epoch - self.stage + 1.)
                tmp = (self.assign_labels[branch][targets, :] - labels.reshape(-1, 1) == 0).nonzero(as_tuple=False)
                idx = self.label_idx[branch][targets]
                val = self.label_val[branch][targets, idx]
                if len(tmp[:, 0]) > 0:
                    idx[tmp[:, 0]] = tmp[:, 1]
                    val[tmp[:, 0]] = 0.
                self.label_val[branch][targets, idx] -= val
                self.label_val[branch][targets, :] *= (1. - factor) / (1. - val.reshape(-1, 1))
                self.assign_labels[branch][targets, idx] = labels
                self.label_val[branch][targets, idx] += factor
                self.label_idx[branch][targets] = torch.min(self.label_val[branch][targets, :], dim=1).indices

    @torch.no_grad()
    def get_label(self, target, epoch, branch):
        if epoch <= self.stage or self.ls == 1:
            return self.assign_labels[branch][target, 0]
        else:
            labels = torch.zeros(len(target), self.K[branch]).cuda()
            for i, t in enumerate(target):
                labels[i, :].index_add_(0, self.assign_labels[branch][t.item(), :], self.label_val[branch][t.item(), :])
            labels[labels > 0] = torch.exp(labels[labels > 0] / self.t)
            labels /= torch.sum(labels, dim=1, keepdim=True)
            return labels

    @torch.no_grad()
    def update_center(self, epoch):
        if epoch < self.stage:
            for i in range(0, self.num_head):
                self.pre_centers[i] += self.cur_centers[i].clone() - self.pre_centers[i]
        if epoch >= self.stage:
            factor = 1. / (epoch - self.stage + 1.)
            for i in range(0, self.num_head):
                tmp_center = F.normalize(self.cur_centers[i], dim=0) #현재 클러스터 중심 정규화
                self.pre_centers[i] += F.normalize((1. - factor) * self.pre_centers[i] + factor * tmp_center, dim=0) - \
                                       self.pre_centers[i] #이전 클러스터 중심과 현재 클러스터 중심 사이의 보간 수행후, 다시 정규화
                self.cur_centers[i] += self.pre_centers[i].clone() - self.cur_centers[i] #벡터간의 차이
        for i in range(0, self.num_head):
            self.counters[i] = torch.zeros(self.K[i]).cuda()
            print("********************")
            return self.counters[i]

    @torch.no_grad()
    def update_center_mini_batch(self, feats, labels, epoch, branch):
        label_idx, label_count = torch.unique(labels, return_counts=True)
        self.duals[branch][label_idx] -= self.dual_lr / len(labels) * label_count
        self.duals[branch] += self.dual_lr * self.lb[branch]
        if self.ratio < 1:
            self.duals[branch][self.duals[branch] < 0] = 0
        alpha = self.counters[branch][label_idx].float()
        self.counters[branch][label_idx] += label_count
        # print(self.counters[branch][label_idx])
        alpha = alpha / self.counters[branch][label_idx].float()
        self.cur_centers[branch][:, label_idx] = self.cur_centers[branch][:, label_idx] * alpha
        self.cur_centers[branch].index_add_(1, labels, feats.data.T * (1. / self.counters[branch][labels]))
        if epoch < self.stage:
            self.cur_centers[branch][:, label_idx] = F.normalize(self.cur_centers[branch][:, label_idx], dim=0)

    def forward(self, img, target, epoch):
        x = self.encoder(img)
        x_pred = self.predictor(x)
        x_pred = F.normalize(x_pred, dim=1)
        x_proj = F.normalize(x, dim=1)

        pred_view = []
        proj_view = []

        for i in range(0, self.num_head):
            proj_view.append(x_proj.matmul(self.pre_centers[i]) / self.T)
            pred_view.append(x_pred.matmul(self.pre_centers[i]) / self.T)

        with torch.no_grad():
            targets = target
            feats = x_proj
            cur_labels = []
            if epoch == 0:
                for j in range(0, self.num_head):
                    labels = self.gen_label(feats, epoch, j)
                    self.update_label(targets, labels, epoch, j)
                    self.update_center_mini_batch(feats, labels, epoch, j)
                    cur_labels.append(self.get_label(target, epoch, j))
            else:
                for j in range(0, self.num_head):
                    cur_labels.append(self.get_label(target, epoch, j))
                    labels = self.gen_label(feats, epoch, j)
                    self.update_center_mini_batch(feats, labels, epoch, j)
                    self.update_label(targets, labels, epoch, j)
        return pred_view, proj_view, cur_labels

def main_worker(args):
    args = args
    # create model
    assert (args.coke_num_head == len(args.coke_k))
    print("=> creating model '{}'".format(args.arch))
    model = CoKe(
        base_encoder=models.__dict__[args.arch],
        K=args.coke_k,
        dim=args.coke_dim,
        num_ins=args.coke_num_ins,
        num_head=args.coke_num_head,
        T=args.coke_t,
        stage=args.coke_stage,
        t=args.coke_tt,
        dual_lr=args.coke_dual_lr,
        ratio=args.coke_ratio,
        ls=args.coke_ls
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    #model = torch.nn.parallel.DataParallel(model).cuda()
    model.cuda()
    model.load_param()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(params =model.parameters(), lr =args.lr,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            # Map model to be loaded to specified single gpu.
            loc = 'cuda'
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    aug = [
        transforms.RandomResizedCrop(32, scale=(0.3, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform = SingleCropsTransform(transforms.Compose(aug)))  #single randomcrop


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False)
    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, scaler)
        model.update_center(epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename='model/{}_{:04d}.pth.tar'.format(args.log, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    a_top1 = AverageMeter('aAcc@1', ':6.2f')
    a_top5 = AverageMeter('aAcc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, a_top1, a_top5, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    train_loader_len = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args, i, train_loader_len)
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        # compute loss
        target = target.cuda()
        with autocast():
            pred_view, proj_view, labels = model(images, target, epoch)
            loss_pred = 0
            loss_proj = 0
            if epoch <= args.coke_stage or args.coke_ls == 1:
                for j in range(0, args.coke_num_head):
                    loss_pred += criterion(pred_view[j], labels[j])
                    loss_proj += criterion(proj_view[j], labels[j])
                label_comp = labels[0]  #target값
            else:
                for j in range(0, args.coke_num_head):
                    loss_pred -= torch.mean(torch.sum(F.log_softmax(pred_view[j], dim=1) * labels[j], dim=1))
                    loss_proj -= torch.mean(torch.sum(F.log_softmax(proj_view[j], dim=1) * labels[j], dim=1))
                label_comp = torch.max(labels[0], dim=1).indices
            loss = args.coke_beta * loss_pred / args.coke_num_head + (
                        1. - args.coke_beta) * loss_proj / args.coke_num_head  #coke_beta

            #print("---------------loss--------------------------",loss)
        a_acc1, a_acc5 = accuracy(pred_view[0], label_comp, topk=(1, 5))
        a_top1.update(a_acc1[0], images[0].size(0))
        a_top5.update(a_acc5[0], images[0].size(0))
        losses.update(loss.item(), images[0].size(0))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(train_loader_len)
    for i in range(0, args.coke_num_head):
        print('max and min cluster size for {}-class clustering is ({},{})'.format(args.coke_k[i], torch.max(
            model.counters[i].data).item(), torch.min(model.counters[i].data).item()))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if (state['epoch'] - 1) % 10 != 0 or state['epoch'] == 1:
        return
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, iteration, num_iter):
    warmup_epoch = 11
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter
    lr = args.lr * (1. + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < warmup_epoch:
        lr = args.lr * max(1, current_iter - num_iter) / (warmup_iter - num_iter)
    if epoch == 0:
        lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # print("lr: ",lr)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main_worker(args)

"""맨하탄 거리(Manhattan distance),
체비셰프 거리(Chebyshev distance)

"""

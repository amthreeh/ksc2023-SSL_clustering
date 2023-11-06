# -*- coding: utf-8 -*-

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import easydict

from torch.utils.data import Dataset
from torchvision import datasets

from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import torch
import torchvision.models as models

from PIL import Image, ImageFilter, ImageOps
import random
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform


        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(".jpg")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert('RGB')
        label = os.path.basename(image_path).split('.')[0]
        label = int(label[0])
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)

        return image, label

args = easydict.EasyDict({
    'arch': 'resnet18',
    "workers" : 0, #현재 시스템 최대 병렬로 작업자 수
    "epochs" : 90,
    "start_epoch":0,
    "batch_size": 128,
    "lr": 0.2,
    "momentum": 0.9,
    "wd" : 1e-6,
    "weight_decay": 1e-6,
    "p" : 100,
    "print_freq": 100,
    "resume": '',
    "evaluate" : 'evaluate',
    "pretrained": "model/cok_s200_c10_0200.pth.tar"})

best_acc1 = 0
def main_worker(args):

    global best_acc1
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.resnet18()
    linear_keyword = 'fc'

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            # print(name, param)
            # print("삭제")
            param.requires_grad = False
    # init the fc layer
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder') and not k.startswith('encoder.%s' % linear_keyword):
                    state_dict[k[len("encoder."):]] = state_dict[k]
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
            print("=> loaded pre-trained model '{}'".format(args.pretrained))

        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))


    # model = torch.nn.parallel.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda() # define loss function (criterion) and optimizer

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # weight, bias

    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            loc = 'cuda'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# CIFAKE Dataset    
    # root="/data"
    # train_dataset=CustomDataset(root, transform = transforms.Compose([
    #                 transforms.RandomResizedCrop(32),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 normalize,
    #         ])
    #     )   
    # valid_dataset = CustomDataset(root, transform = transforms.Compose([
    #         transforms.Resize(32),
    #         transforms.CenterCrop(28),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    #    )
    
# STL-10 Dataset
    # train_dataset = datasets.STL10(
    #     root="data",
    #     split='train',
    #     download=False,
    #     transform = transforms.Compose([
    #                 transforms.RandomResizedCrop(96),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 normalize,
    #                 ])
    #     )
    # valid_dataset = datasets.STL10(
    #     root="data",
    #     split='test',
    #     download=False,
    #     transform = transforms.Compose([
    #         transforms.Resize(96),
    #         transforms.CenterCrop(84),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    #    )

# CIFAR10 Dataset
    train_dataset = datasets.CIFAR10(
        root="data/dhk",
        train=True,
        download=True,
        transform = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    ])
        )
    valid_dataset = datasets.CIFAR10(
        root="data/dhk",
        train=False,
        download=True,
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            normalize,
        ])
       )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc1, val_losses = validate(val_loader, model, criterion, epoch, args)
        #test_acc1, test_losses = test(test_loader, model, criterion, epoch, args)

        val_acc_list.append(val_acc1)
        #test_acc_list.append(test_acc1)
        val_loss_list.append(val_losses)
        #test_loss_list.append(test_losses)

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        if epoch == args.start_epoch:
            sanity_check(model.state_dict(), args.pretrained, linear_keyword)
        print("best", best_acc1)
 

    if args.evaluate:
        validate(val_loader, model, criterion, epoch, args)

        return

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    model = model.cuda()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
     
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    


def validate(val_loader, model, criterion, epoch, args):
    actual = []
    deep_features = []

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            deep_features += output.cpu().numpy().tolist()
            actual += target.cpu().numpy().tolist()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMester
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    #2차원 축소
    # tsne = TSNE(n_components=2, random_state=0)
    # train_cluster = np.array(tsne.fit_transform(np.array(deep_features)))
    # train_actual = np.array(actual)

    # # clustering 시각화
    # plt.figure(figsize=(10, 10))
    # stl = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    # for i, label in zip(range(10), stl):
    #     idx = np.where(train_actual == i)
    #     plt.scatter(train_cluster[idx, 0], train_cluster[idx, 1], marker='.', label=label)
    # plt.title(f"cc_single_view{epoch}")
    # plt.legend()
    # # plt.plot()
    # plt.savefig(f'cc_single_c10/s180_{epoch}.png')

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']


    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue
        # name in pretrained model
        # k_pre = 'module.encoder.' + k[len('module.'):] \
        #     if k.startswith('module.') else 'module.encoder.' + k
        # name in pretrained model
        k_pre = 'encoder.' + k[len('.'):] \
            if k.startswith('.') else 'encoder.' + k

        #print("k_pre", k_pre)
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


    #         # name in pretrained model
    #     k_pre = 'encoder.' + (k[1:] if k.startswith('.') else k)

    #     assert torch.allclose(state_dict[k], state_dict_pre[k_pre]), \
    #         '{} is changed in linear classifier training.'.format(k)

    # print("=> sanity check passed.")


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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


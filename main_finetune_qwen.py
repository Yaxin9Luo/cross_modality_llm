# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from torchvision import datasets, transforms
from transformers import AutoConfig, AutoModel
from timm.models.vision_transformer import PatchEmbed, Block
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler, NoOpScaler
from engine_finetune import train_one_epoch, evaluate
import torch.nn as nn
from model_qwen import MAE_Qwen3_Classifier  # Changed from model_llm
import logging

def get_args_parser():
    parser = argparse.ArgumentParser('Qwen3-8B fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--random_label_ratio', default=0.0, type=float,
                        help='Ratio of random labels in the training set')
    # Model parameters
    parser.add_argument('--model', default='qwen3-8b', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrained_model_path', type=str, default='ckpts/Qwen3-8B',
                        help='Path to pretrained Qwen3-8B checkpoint')
    parser.add_argument('--random_init', action='store_true',
                        help='Use random initialization instead of pretrained weights')
    parser.add_argument('--load_pretrained', action='store_true',
                        help='Load pretrained Qwen3-8B weights')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./eval_output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./eval_output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--linear_probe', action='store_true',
                        help='Only train the classifier layer (linear probing)')
    parser.add_argument('--qwen_checkpoint', type=str, default='',
                        help='Path to fine-tuned Qwen3 checkpoint (for resuming training)')
    # Dataset selection parameter
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet'],
                        help='Dataset to use for training')
    return parser

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            print(f"Non-trainable param: {name}, Shape: {param.shape}")
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
def build_imagenet_dataset(is_train, args):
    transform = build_imagenet_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root=root, transform=transform)
    return dataset
def build_imagenet_dataset_with_random_labels(is_train, args, random_label_ratio=1.0):
    dataset = build_imagenet_dataset(is_train, args)
    
    if is_train:
        random.seed(args.seed)
        num_classes = args.nb_classes
        num_samples = len(dataset)
        num_random_samples = int(num_samples * random_label_ratio)
        
        # 首先将所有标签映射到0-(num_classes-1)范围内
        dataset.targets = [label % num_classes for label in dataset.targets]
        
        random_indices = random.sample(range(num_samples), num_random_samples)
        
        print("\n正在为训练集随机化标签:")
        for i, idx in enumerate(random_indices):
            original_label = dataset.targets[idx]
            new_label = random.randint(0, num_classes - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_classes - 1)
            dataset.targets[idx] = new_label
            
            if i < 10:
                print(f"样本 {idx}: 原始标签: {original_label}, 新标签: {new_label}")
            elif i == 10:
                print("...")
        
        print(f"\n在训练集中的 {num_samples} 个样本中随机化了 {num_random_samples} 个标签。")
        
        label_counts = {i: 0 for i in range(num_classes)}
        for label in dataset.targets:
            label_counts[label] += 1
        print("\n随机化后的标签分布:")
        for label, count in sorted(label_counts.items())[:10]:
            print(f"标签 {label}: {count} 个样本")
        print("...")
    
    return dataset
def build_cifar_dataset(is_train, args):
    transform = build_cifar_transform(is_train, args)
    try:
        # 第一次尝试正常下载
        dataset = datasets.CIFAR10(root=args.data_path, train=is_train, transform=transform, download=True)
    except Exception as e:
        print(f"首次下载尝试失败: {str(e)}")
        try:
            # 第二次尝试：禁用SSL验证
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            dataset = datasets.CIFAR10(root=args.data_path, train=is_train, transform=transform, download=True)
        except Exception as e:
            print(f"第二次下载尝试失败: {str(e)}")
            # 如果数据集已经存在，尝试直接加载
            try:
                dataset = datasets.CIFAR10(root=args.data_path, train=is_train, transform=transform, download=False)
                print("成功从本地加载数据集")
            except Exception as e:
                raise Exception(f"无法加载CIFAR10数据集: {str(e)}")
    
    return dataset
def build_cifar_dataset_with_random_labels(is_train, args, random_label_ratio=1.0):
    dataset = build_cifar_dataset(is_train, args)
    if is_train:
        # random_seed = args.seed + misc.get_rank() # GPUs
        random.seed(args.seed)
        num_classes = 10  # CIFAR-10 has 10 classes
        num_samples = len(dataset)
        num_random_samples = int(num_samples * random_label_ratio)
        
        random_indices = random.sample(range(num_samples), num_random_samples)
        
        print("\nRandomizing labels for training set:")
        for i, idx in enumerate(random_indices):
            original_label = dataset.targets[idx]
            new_label = random.randint(0, num_classes - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_classes - 1)
            dataset.targets[idx] = new_label
            
            if i < 10:  # Print first 10 changes for verification
                print(f"Sample {idx}: Original label: {original_label}, New label: {new_label}")
            elif i == 10:
                print("...")
        
        print(f"\nRandomized {num_random_samples} labels out of {num_samples} in the training set.")
        
        # Verify the overall distribution of labels
        label_counts = {i: 0 for i in range(num_classes)}
        for label in dataset.targets:
            label_counts[label] += 1
        print("\nLabel distribution after randomization:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} samples")
    
    return dataset
def build_cifar_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    return transform
def build_imagenet_transform(is_train, args):
    """
    构建用于ImageNet训练和验证的标准数据增强流程
    参考：
    1. PyTorch官方实现
    2. TimM库的最佳实践
    3. ResNet、ViT等经典论文中的数据增强策略
    """
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size, 
                scale=(0.08, 1.0),  
                ratio=(3./4., 4./3.),  
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if args.aa else transforms.Lambda(lambda x: x),
            
            transforms.ColorJitter(
                brightness=0.4 * args.color_jitter if args.color_jitter is not None else 0,
                contrast=0.4 * args.color_jitter if args.color_jitter is not None else 0,
                saturation=0.4 * args.color_jitter if args.color_jitter is not None else 0,
                hue=0.2 * args.color_jitter if args.color_jitter is not None else 0,
            ) if args.color_jitter is not None else transforms.Lambda(lambda x: x),
            
            transforms.ToTensor(),
            
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            transforms.RandomErasing(
                p=args.reprob,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
            ) if args.reprob > 0 else transforms.Lambda(lambda x: x),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(
                int(args.input_size / 0.875),  
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    return transform
def setup_logging(args):
    if misc.get_rank() == 0:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            log_file = os.path.join(args.output_dir, f'train_rank{misc.get_rank()}.log')
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
    else:
        logging.basicConfig(level=logging.ERROR)
def save_checkpoint(args, epoch, model, optimizer, loss_scaler, is_best=False):
    if misc.get_rank() == 0:  # 只在主进程保存
        save_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
        
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(save_dict, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(args.output_dir, 'model_best.pth')
            torch.save(save_dict, best_path)
def build_cifar100_dataset(is_train, args):
    transform = build_cifar_transform(is_train, args)
    try:
        # 第一次尝试正常下载
        dataset = datasets.CIFAR100(root=args.data_path, train=is_train, transform=transform, download=True)
    except Exception as e:
        print(f"首次下载尝试失败: {str(e)}")
        try:
            # 第二次尝试：禁用SSL验证
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            dataset = datasets.CIFAR100(root=args.data_path, train=is_train, transform=transform, download=True)
        except Exception as e:
            print(f"第二次下载尝试失败: {str(e)}")
            # 如果数据集已经存在，尝试直接加载
            try:
                dataset = datasets.CIFAR100(root=args.data_path, train=is_train, transform=transform, download=False)
                print("成功从本地加载数据集")
            except Exception as e:
                raise Exception(f"无法加载CIFAR100数据集: {str(e)}")
    
    return dataset
def build_cifar100_dataset_with_random_labels(is_train, args, random_label_ratio=1.0):
    dataset = build_cifar100_dataset(is_train, args)
    if is_train:
        random.seed(args.seed)
        num_classes = 100  # CIFAR-100 has 100 classes
        num_samples = len(dataset)
        num_random_samples = int(num_samples * random_label_ratio)
        
        random_indices = random.sample(range(num_samples), num_random_samples)
        
        print("\n正在为训练集随机化标签:")
        for i, idx in enumerate(random_indices):
            original_label = dataset.targets[idx]
            new_label = random.randint(0, num_classes - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_classes - 1)
            dataset.targets[idx] = new_label
            
            if i < 10:
                print(f"样本 {idx}: 原始标签: {original_label}, 新标签: {new_label}")
            elif i == 10:
                print("...")
        
        print(f"\n在训练集中的 {num_samples} 个样本中随机化了 {num_random_samples} 个标签。")
        
        label_counts = {i: 0 for i in range(num_classes)}
        for label in dataset.targets:
            label_counts[label] += 1
        print("\n随机化后的标签分布:")
        for label, count in sorted(label_counts.items())[:10]:
            print(f"标签 {label}: {count} 个样本")
        print("...")
    
    return dataset
def build_tiny_imagenet_dataset(is_train, args):
    transform = build_imagenet_transform(is_train, args)  # 使用与ImageNet相同的transform
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root=root, transform=transform)
    return dataset
def build_tiny_imagenet_dataset_with_random_labels(is_train, args, random_label_ratio=1.0):
    dataset = build_tiny_imagenet_dataset(is_train, args)
    
    if is_train and random_label_ratio > 0.0:  # 只在需要随机化标签时执行
        random.seed(args.seed)
        num_classes = 200  # TinyImageNet有200个类别
        num_samples = len(dataset)
        num_random_samples = int(num_samples * random_label_ratio)
        
        # 首先将所有标签映射到0-(num_classes-1)范围内
        dataset.targets = [label % num_classes for label in dataset.targets]
        
        random_indices = random.sample(range(num_samples), num_random_samples)
        
        print("\n正在为训练集随机化标签:")
        for i, idx in enumerate(random_indices):
            original_label = dataset.targets[idx]
            new_label = random.randint(0, num_classes - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_classes - 1)
            dataset.targets[idx] = new_label
            
            if i < 10:
                print(f"样本 {idx}: 原始标签: {original_label}, 新标签: {new_label}")
            elif i == 10:
                print("...")
        
        print(f"\n在训练集中的 {num_samples} 个样本中随机化了 {num_random_samples} 个标签。")
        
        label_counts = {i: 0 for i in range(num_classes)}
        for label in dataset.targets:
            label_counts[label] += 1
        print("\n随机化后的标签分布:")
        for label, count in sorted(label_counts.items())[:10]:
            print(f"标签 {label}: {count} 个样本")
        print("...")
    elif is_train:
        print("\nCorrect labels Training!")
    
    return dataset
def main(args):
    misc.init_distributed_mode(args)

    print(f'Starting main function on rank {misc.get_rank()}')
    
    # 添加同步点
    if args.distributed:
        torch.distributed.barrier()
        print(f'Passed first barrier on rank {misc.get_rank()}')
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print(f"World size: {misc.get_world_size()}")
    print(f"Current rank: {misc.get_rank()}")
    print(f"Local rank: {args.local_rank}")

    device = torch.device(args.device)

    # 设置每个进程的随机种子
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True



    # dataset_train = build_dataset(is_train=True, args=args)
    # dataset_val = build_dataset(is_train=False, args=args)
    # 根据选择的数据集加载相应的数据
    if args.dataset == 'cifar10':
        dataset_train = build_cifar_dataset_with_random_labels(is_train=True, args=args, random_label_ratio=args.random_label_ratio)
        dataset_val = build_cifar_dataset(is_train=False, args=args)
        args.nb_classes = 10
    elif args.dataset == 'cifar100':
        dataset_train = build_cifar100_dataset_with_random_labels(is_train=True, args=args, random_label_ratio=args.random_label_ratio)
        dataset_val = build_cifar100_dataset(is_train=False, args=args)
        args.nb_classes = 100
    elif args.dataset == 'imagenet':
        dataset_train = build_imagenet_dataset_with_random_labels(is_train=True, args=args, random_label_ratio=args.random_label_ratio)
        dataset_val = build_imagenet_dataset(is_train=False, args=args)
        args.nb_classes = 1000
    elif args.dataset == 'tiny-imagenet':
        dataset_train = build_tiny_imagenet_dataset_with_random_labels(is_train=True, args=args, random_label_ratio=args.random_label_ratio)
        dataset_val = build_tiny_imagenet_dataset(is_train=False, args=args)
        args.nb_classes = 200
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    # Create the Qwen3-8B model
    # Determine whether to load pretrained weights
    use_pretrained = args.load_pretrained and not args.random_init

    print(f"Creating Qwen3-8B model...")
    print(f"  - Pretrained: {use_pretrained}")
    print(f"  - Linear probe: {args.linear_probe}")
    print(f"  - Gradient checkpointing: {args.gradient_checkpointing}")

    model = MAE_Qwen3_Classifier(
        args,
        pretrained=use_pretrained,
        linear_probe=args.linear_probe,
        model_path=args.pretrained_model_path
    )

    # Load from checkpoint if resuming training
    if args.qwen_checkpoint:
        print(f"Loading weights from {args.qwen_checkpoint}")
        checkpoint = torch.load(args.qwen_checkpoint, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']

            # Separate Qwen3 and classifier weights
            qwen_state_dict = {k[5:]: v for k, v in state_dict.items()
                             if k.startswith('qwen.')}

            # If linear probe mode, only load Qwen3 backbone
            if args.linear_probe:
                print("Linear probing mode: Loading only Qwen3 weights")
                model.qwen.load_state_dict(qwen_state_dict, strict=False)
                # Re-initialize classifier
                model._init_weights()
            else:
                # Full fine-tuning mode: load all weights
                print("Full fine-tuning mode: Loading all weights")
                model.load_state_dict(state_dict, strict=False)

            print("Successfully loaded weights from checkpoint")
    model.to(device)

    model_without_ddp = model
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (total_parameters / 1.e6))
    print('number of trainable params (M): %.2f' % (trainable_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[args.gpu],
                                                        find_unused_parameters=True)
        model_without_ddp = model.module
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) ##### Add scheduler

    # Use NoOpScaler for bfloat16 (GradScaler doesn't support bfloat16)
    loss_scaler = NoOpScaler()
    print("Using NoOpScaler for bfloat16 training (no gradient scaling)")

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {args.start_epoch}")
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        scheduler.step() ##### Add scheduler
        if args.output_dir and (epoch + 1) % 10 == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': total_parameters,
                        'n_trainable_parameters': trainable_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(args)
    main(args)

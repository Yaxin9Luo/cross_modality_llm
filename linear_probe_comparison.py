# Copyright (c) 2026
# Unified Linear Probing Comparison: LBBT (GPT-2) vs DINO
# --------------------------------------------------------
# Compares cross-modality transfer (LBBT) against standard vision SSL (DINO)
# using linear probing evaluation protocol.
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS

from model_llm import MAE_GPT2_Classifier
from models_dino import build_dino_model, AVAILABLE_MODELS


def accuracy(output, target, topk=(1,)):
    """Compute accuracy over the k top predictions."""
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


def get_args_parser():
    parser = argparse.ArgumentParser('LBBT vs DINO Linear Probing Comparison', add_help=False)
    
    # Common parameters
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # Model selection
    parser.add_argument('--method', default='lbbt', type=str, 
                        choices=['lbbt', 'dino', 'dinov2'],
                        help='Method to evaluate: lbbt (GPT-2), dino, or dinov2')
    parser.add_argument('--dino_model', default='dinov2_vitl14', type=str,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='DINO/DINOv2 model variant (default: dinov2_vitl14 for fair comparison with GPT-2 Medium)')
    parser.add_argument('--lbbt_checkpoint', type=str, default='',
                        help='Path to pretrained LBBT (GPT-2) checkpoint')
    parser.add_argument('--lbbt_pretrained', action='store_true',
                        help='Use language-pretrained GPT-2 for LBBT')
    
    # Input parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='Image input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (default: 0 for linear probe)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet100', 'imagenet', 'tiny-imagenet'],
                        help='Dataset to use')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='Dataset path')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='Number of classification classes')

    # Output parameters
    parser.add_argument('--output_dir', default='./linprobe_comparison_results',
                        help='Path where to save results')
    parser.add_argument('--log_dir', default='./linprobe_comparison_results',
                        help='Path for tensorboard logs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    
    # Distributed training parameters
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_eval', action='store_true', default=False)

    return parser


def build_cifar_transform(is_train, input_size=224):
    """Build transforms for CIFAR datasets."""
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875), interpolation=3),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


def build_imagenet_transform(is_train, input_size=224):
    """Build transforms for ImageNet-style datasets."""
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875), interpolation=3),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


def build_dataset(is_train, args):
    """Build dataset based on args.dataset."""
    if args.dataset == 'cifar10':
        transform = build_cifar_transform(is_train, args.input_size)
        dataset = datasets.CIFAR10(root=args.data_path, train=is_train, 
                                   transform=transform, download=True)
        args.nb_classes = 10
    elif args.dataset == 'cifar100':
        transform = build_cifar_transform(is_train, args.input_size)
        dataset = datasets.CIFAR100(root=args.data_path, train=is_train,
                                    transform=transform, download=True)
        args.nb_classes = 100
    elif args.dataset in ['imagenet', 'imagenet100', 'tiny-imagenet']:
        transform = build_imagenet_transform(is_train, args.input_size)
        split = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(os.path.join(args.data_path, split), transform=transform)
        if args.dataset == 'imagenet100':
            args.nb_classes = 100
        elif args.dataset == 'tiny-imagenet':
            args.nb_classes = 200
        else:
            args.nb_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return dataset


def build_model(args):
    """Build model based on method selection."""
    if args.method == 'lbbt':
        print(f"Building LBBT (GPT-2) model...")
        model = MAE_GPT2_Classifier(args, pretrained=args.lbbt_pretrained, linear_probe=True)
        
        if args.lbbt_checkpoint:
            print(f"Loading LBBT checkpoint from: {args.lbbt_checkpoint}")
            checkpoint = torch.load(args.lbbt_checkpoint, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                # Load GPT2 weights only (skip classifier)
                gpt2_state_dict = {k[5:]: v for k, v in state_dict.items() 
                                   if k.startswith('gpt2.')}
                patch_embed_state_dict = {k[12:]: v for k, v in state_dict.items()
                                          if k.startswith('patch_embed.')}
                model.gpt2.load_state_dict(gpt2_state_dict)
                model.patch_embed.load_state_dict(patch_embed_state_dict)
                print("Loaded GPT2 and patch_embed weights from checkpoint")
                # Re-initialize classifier
                model.initialize_patch_embed_and_classifier()
            else:
                model.load_state_dict(checkpoint, strict=False)
        
        # Freeze all but classifier for linear probe
        for param in model.gpt2.parameters():
            param.requires_grad = False
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif args.method in ['dino', 'dinov2']:
        print(f"Building {args.method.upper()} model: {args.dino_model}")
        model = build_dino_model(args, args.dino_model, pretrained=True, linear_probe=True)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    return model


def get_trainable_params(model, args):
    """Get trainable parameters for optimizer."""
    if args.method == 'lbbt':
        return model.classifier.parameters()
    else:  # DINO
        return model.classifier.parameters()


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, 
                    loss_scaler, max_norm=None, log_writer=None, args=None):
    """Train for one epoch."""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Adjust learning rate per iteration
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        loss_value = loss.item()
        
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            raise ValueError(f"Loss is {loss_value}")
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        # Compute accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(acc1=acc1.item())
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/lr', lr, epoch_1000x)
            log_writer.add_scalar('train/acc1', acc1.item(), epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """Evaluate model on validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    model.eval()
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with cosine schedule."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def main(args):
    misc.init_distributed_mode(args)
    
    print('=' * 60)
    print(f'Linear Probing Comparison: {args.method.upper()}')
    print(f'Dataset: {args.dataset}')
    if args.method in ['dino', 'dinov2']:
        print(f'Model: {args.dino_model}')
    print('=' * 60)
    
    device = torch.device(args.device)
    
    # Fix seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    cudnn.benchmark = True
    
    # Build datasets
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    print(f"Train dataset: {len(dataset_train)} samples")
    print(f"Val dataset: {len(dataset_val)} samples")
    print(f"Number of classes: {args.nb_classes}")
    
    # Build samplers
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        global_rank = 0
    
    # Setup logging
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    # Build data loaders
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
    
    # Build model
    model = build_model(args)
    model.to(device)
    
    model_without_ddp = model
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.4f}M")
    
    # Compute effective batch size and learning rate
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")
    
    # Distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # Optimizer - LARS for linear probing (following MoCo v3)
    optimizer = LARS(get_trainable_params(model_without_ddp, args), 
                     lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: {optimizer}")
    
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
            print(f"Resumed from epoch {args.start_epoch}")
    
    # Evaluation only
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of {args.method.upper()} on {args.dataset}: {test_stats['acc1']:.2f}%")
        return
    
    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None, log_writer=log_writer, args=args
        )
        
        # Save checkpoint
        if args.output_dir and (epoch + 1) % 10 == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )
        
        # Evaluate
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of {args.method.upper()} on {args.dataset}: {test_stats['acc1']:.2f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        # Log
        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'method': args.method,
            'dataset': args.dataset,
            'max_accuracy': max_accuracy,
        }
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # Final summary
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print('=' * 60)
    print(f'Training completed for {args.method.upper()} on {args.dataset}')
    print(f'Total training time: {total_time_str}')
    print(f'Best accuracy: {max_accuracy:.2f}%')
    print('=' * 60)
    
    # Save final results
    if args.output_dir and misc.is_main_process():
        results = {
            'method': args.method,
            'dataset': args.dataset,
            'model': args.dino_model if args.method in ['dino', 'dinov2'] else 'gpt2-medium',
            'best_acc1': max_accuracy,
            'final_acc1': test_stats['acc1'],
            'final_acc5': test_stats['acc5'],
            'epochs': args.epochs,
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6,
            'training_time': total_time_str,
        }
        with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    if args.output_dir:
        # Create output directory with method and dataset in name
        args.output_dir = os.path.join(
            args.output_dir, 
            f"{args.method}_{args.dataset}_{args.dino_model if args.method in ['dino', 'dinov2'] else 'gpt2'}"
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.log_dir = args.output_dir
    
    main(args)

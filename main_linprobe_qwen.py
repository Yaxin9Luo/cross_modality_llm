#!/usr/bin/env python
# Linear probing evaluation for fine-tuned Qwen3-8B models
# Loads a fine-tuned checkpoint, freezes the backbone, and trains only the classifier

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler, NoOpScaler
from util.lars import LARS
from model_qwen import MAE_Qwen3_Classifier
from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Qwen3-8B Linear Probing Evaluation', add_help=False)

    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='qwen3-8b', type=str,
                        help='Model name')
    parser.add_argument('--pretrained_model_path', type=str, default='ckpts/Qwen3-8B',
                        help='Path to Qwen3-8B model files')
    parser.add_argument('--finetune_checkpoint', type=str, required=True,
                        help='Path to fine-tuned checkpoint to evaluate')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Image input size')
    parser.add_argument('--linear_probe', action='store_true', default=True,
                        help='Linear probing mode (freeze backbone)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing (not needed for linear probe)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1,
                        help='Base learning rate: absolute_lr = base_lr * batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='Lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet'],
                        help='Dataset to use')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='Dataset path')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='Number of classification classes')

    # Output parameters
    parser.add_argument('--output_dir', default='./linprobe_results',
                        help='Path where to save results')
    parser.add_argument('--log_dir', default='./linprobe_results',
                        help='Path for tensorboard logs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
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


def build_cifar_dataset(is_train, args):
    """Build CIFAR-10/100 dataset."""
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(args.input_size / 0.875)),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_path, train=is_train, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root=args.data_path, train=is_train, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")

    return dataset


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Build dataset
    print(f"Building {args.dataset} dataset...")
    dataset_train = build_cifar_dataset(True, args)
    dataset_val = build_cifar_dataset(False, args)

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

    # Create model with linear probe mode
    print(f"Loading fine-tuned model from {args.finetune_checkpoint}")

    # First create model structure (no pretrained weights needed, we'll load from checkpoint)
    model = MAE_Qwen3_Classifier(
        args,
        pretrained=False,  # Don't load pretrained, we'll load from checkpoint
        linear_probe=True,  # Freeze backbone
        model_path=args.pretrained_model_path
    )

    # Load fine-tuned checkpoint
    checkpoint = torch.load(args.finetune_checkpoint, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Load the full state dict from fine-tuning
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine-tuned weights from {args.finetune_checkpoint}")

        # Re-freeze backbone (in case it wasn't frozen)
        for param in model.qwen.parameters():
            param.requires_grad = False

        # Re-initialize classifier for fresh linear probing
        model._init_weights()
        print("Re-initialized classifier for linear probing")

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Use LARS optimizer for linear probing (following MoCo v1/v3)
    optimizer = LARS(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)

    # Use NoOpScaler for bfloat16 (GradScaler doesn't support bfloat16)
    loss_scaler = NoOpScaler()
    print("Using NoOpScaler for bfloat16 training (no gradient scaling)")

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start linear probing for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,  # No gradient clipping for linear probe
            log_writer=log_writer,
            args=args
        )

        if args.output_dir:
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
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Save final results
    if args.output_dir and misc.is_main_process():
        final_results = {
            'best_acc1': max_accuracy,
            'final_acc1': test_stats['acc1'],
            'final_acc5': test_stats['acc5'],
            'total_epochs': args.epochs,
            'total_time': total_time_str
        }
        with open(os.path.join(args.output_dir, "final_results.json"), mode="w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)

        print("\n" + "="*50)
        print("Linear Probing Evaluation Complete!")
        print(f"Best Accuracy: {max_accuracy:.2f}%")
        print(f"Final Accuracy: {test_stats['acc1']:.2f}%")
        print("="*50)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

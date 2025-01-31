# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import torch.nn.functional as F  

import torch
import wandb
from timm.data import Mixup
from timm.utils import accuracy
from collections import defaultdict

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc1', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # add acc1
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # # 添加梯度相关的meters
    # for i in range(24):  # 假设有12个transformer blocks
    #     metric_logger.add_meter(f'grad_norm_block_{i}', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #     metric_logger.add_meter(f'grad_cos_sim_block_{i}', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # # 存储前一次的梯度
    # previous_block_grads = {}
    
    def compute_cosine_similarity(grad1, grad2):
        """计算两个梯度向量的余弦相似度"""
        if grad1 is None or grad2 is None:
            return 0.0
        return float(F.cosine_similarity(grad1.view(1, -1), grad2.view(1, -1)).cpu())
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    def log_block_gradients(step):
        grad_stats = {}
        block_grads = {}

        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
                
            if 'gpt2.h.' in name:
                try:
                    block_num = int(name.split('gpt2.h.')[1].split('.')[0])
                    if block_num not in block_grads:
                        block_grads[block_num] = []
                    grad_flat = param.grad.detach().flatten()
                    block_grads[block_num].append(grad_flat)
                except Exception as e:
                    print(f"Error processing {name}: {e}")
        
        # 计算并记录每个block的统计信息
        for block_num, grads in block_grads.items():
            try:
                # 合并当前block的所有梯度
                current_block_grads = torch.cat(grads)
                
                # 计算梯度范数
                grad_norm = float(torch.norm(current_block_grads, p=2).cpu())
                
                # 计算与前一次梯度的余弦相似度
                cos_sim = 0.0
                if block_num in previous_block_grads:
                    cos_sim = compute_cosine_similarity(
                        current_block_grads,
                        previous_block_grads[block_num]
                    )
                
                # 更新metric logger
                metric_logger.update(**{
                    f'grad_norm_block_{block_num}': grad_norm,
                    f'grad_cos_sim_block_{block_num}': cos_sim
                })
                
                # 记录到wandb
                grad_stats.update({
                    f"grad/block_{block_num}/l2_norm": grad_norm,
                    f"grad/block_{block_num}/cos_sim": cos_sim,
                    f"grad/block_{block_num}/mean": float(torch.mean(current_block_grads).cpu()),
                    f"grad/block_{block_num}/std": float(torch.std(current_block_grads).cpu())
                })
                
                # 保存当前梯度用于下次比较
                previous_block_grads[block_num] = current_block_grads.detach().clone()
                
            except Exception as e:
                print(f"Error computing stats for block {block_num}: {e}")
        if grad_stats and wandb.run is not None:
            wandb.log(grad_stats, step=epoch * len(data_loader) + step)
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        metric_logger.update(acc1=acc1.item())

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            # log_block_gradients(data_iter_step)
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # if (data_iter_step + 1) % accum_iter == 0:
        #     # 记录基本训练指标到W&B
        #     wandb.log({
        #         "loss": loss_value_reduce,
        #         "learning_rate": max_lr,
        #         "acc1": acc1.item(),
        #         "epoch": epoch + data_iter_step / len(data_loader)
        #     }, step=epoch * len(data_loader) + data_iter_step)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            # # 添加梯度信息到log_writer
            # for name, meter in metric_logger.meters.items():
            #     if name.startswith('grad_'):
            #         log_writer.add_scalar(name, meter.global_avg, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# DeepSpeed ZeRO-2 Training Guide for Qwen

This guide explains how to use DeepSpeed ZeRO-2 optimization for training Qwen3-8B models with significantly reduced GPU memory usage.

## Overview

DeepSpeed ZeRO-2 has been integrated into the Qwen training pipeline to enable:
- **50-60% GPU memory reduction** compared to standard DDP
- **Larger batch sizes** for improved training throughput
- **Optional CPU offloading** for extreme memory savings (60-70% reduction)

## Quick Start

### 1. Install DeepSpeed

```bash
pip install deepspeed>=0.12.0
# or with uv
uv pip install deepspeed>=0.12.0
```

### 2. Run Training with DeepSpeed

```bash
# Basic DeepSpeed training (2 GPUs)
deepspeed --num_gpus=2 main_finetune_qwen.py \
    --deepspeed \
    --deepspeed_config deepspeed_configs/zero2_config.json \
    --load_pretrained \
    --pretrained_model_path ./Qwen3-8B \
    --dataset cifar100 \
    --data_path ./data \
    --nb_classes 100 \
    --batch_size 32 \
    --epochs 100 \
    --output_dir ./results/deepspeed_training

# Or use the experiment script
bash qwen_deepspeed_experiment.sh
```

### 3. With CPU Offloading (for limited GPU memory)

```bash
deepspeed --num_gpus=2 main_finetune_qwen.py \
    --deepspeed \
    --deepspeed_config deepspeed_configs/zero2_offload_config.json \
    --batch_size 32 \
    ...
```

## Configuration Files

Two DeepSpeed configurations are provided:

### 1. `deepspeed_configs/zero2_config.json`
- **ZeRO Stage 2**: Partitions optimizer states and gradients
- **bfloat16 precision**: Native bfloat16 support (no gradient scaling)
- **Gradient clipping**: Automatically configured from command-line args
- **Memory savings**: 50-60% reduction vs DDP

### 2. `deepspeed_configs/zero2_offload_config.json`
- **All features from zero2_config.json**, plus:
- **CPU offloading**: Offloads optimizer states to CPU RAM
- **Memory savings**: 60-70% reduction vs DDP
- **Trade-off**: Slightly slower (5-15%) due to CPU-GPU communication

## Command-Line Arguments

New arguments added for DeepSpeed:

```bash
--deepspeed                    # Enable DeepSpeed training
--deepspeed_config PATH        # Path to DeepSpeed config JSON
                               # (default: deepspeed_configs/zero2_config.json)
```

All existing arguments work with DeepSpeed:
- `--gradient_checkpointing`: Stacks with DeepSpeed for maximum memory savings
- `--clip_grad`: Automatically applied in DeepSpeed config
- `--batch_size`: Can be increased with DeepSpeed (e.g., 32 instead of 8)

## Memory Usage Comparison

| Configuration | GPU Memory (per GPU) | Batch Size | Throughput |
|--------------|---------------------|------------|------------|
| Standard DDP | ~40-50 GB | 8 | 100% (baseline) |
| DeepSpeed ZeRO-2 | ~20-25 GB | 32 | ~95% |
| ZeRO-2 + Offload | ~15-20 GB | 32 | ~85-90% |
| ZeRO-2 + GradCkpt | ~15-18 GB | 32 | ~90% |
| ZeRO-2 + Offload + GradCkpt | ~12-15 GB | 32 | ~80-85% |

*Note: Actual memory usage depends on model size, sequence length, and other factors.*

## Checkpointing

### Saving Checkpoints

DeepSpeed saves checkpoints in a different format than standard PyTorch:

```
results/deepspeed_training/
├── checkpoint-10/
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   ├── mp_rank_00_model_states.pt
│   └── client_state.pt
├── checkpoint-20/
│   └── ...
└── checkpoint-final/
    └── ...
```

Checkpoints are automatically saved every 10 epochs and at the end of training.

### Loading Checkpoints

To resume training from a DeepSpeed checkpoint:

```bash
deepspeed --num_gpus=2 main_finetune_qwen.py \
    --deepspeed \
    --deepspeed_config deepspeed_configs/zero2_config.json \
    --resume checkpoint-10 \
    --output_dir ./results/deepspeed_training \
    ...
```

### Converting Checkpoints to Standard PyTorch Format

For evaluation or deployment, convert DeepSpeed checkpoints to standard format:

```bash
python convert_deepspeed_checkpoint.py \
    --checkpoint ./results/deepspeed_training/checkpoint-99 \
    --output ./converted_checkpoint.pth

# Weights only (for inference)
python convert_deepspeed_checkpoint.py \
    --checkpoint ./results/deepspeed_training/checkpoint-99 \
    --output ./model_weights.pth \
    --weights_only
```

## Backward Compatibility

DeepSpeed is **completely optional**. The codebase maintains full backward compatibility:

- **Without `--deepspeed` flag**: Standard DDP training (unchanged behavior)
- **With `--deepspeed` flag**: DeepSpeed ZeRO-2 optimization
- **All existing scripts work**: No changes needed to existing workflows

## Example Workflows

### 1. Full Fine-tuning with DeepSpeed

```bash
# Stage 1: Fine-tune with DeepSpeed (larger batch size)
deepspeed --num_gpus=2 main_finetune_qwen.py \
    --deepspeed \
    --deepspeed_config deepspeed_configs/zero2_config.json \
    --load_pretrained \
    --pretrained_model_path ./Qwen3-8B \
    --dataset cifar100 \
    --batch_size 32 \
    --gradient_checkpointing \
    --epochs 100 \
    --output_dir ./results/qwen_deepspeed

# Stage 2: Convert checkpoint for evaluation
python convert_deepspeed_checkpoint.py \
    --checkpoint ./results/qwen_deepspeed/checkpoint-final \
    --output ./results/qwen_deepspeed/checkpoint-final.pth

# Stage 3: Evaluate with standard DDP
python main_linprobe_qwen.py \
    --finetune_checkpoint ./results/qwen_deepspeed/checkpoint-final.pth \
    --linear_probe \
    --dataset cifar100 \
    --epochs 100 \
    --output_dir ./results/qwen_linear_probe
```

### 2. Comparing DDP vs DeepSpeed

```bash
# Train with standard DDP (baseline)
bash qwen_pretrained_experiment.sh

# Train with DeepSpeed
bash qwen_deepspeed_experiment.sh

# Compare results
diff -y \
    <(tail -n 20 ./results/qwen_pretrained/finetune/log.txt) \
    <(tail -n 20 ./results/qwen_deepspeed/finetune/log.txt)
```

## Troubleshooting

### Issue: "Default process group has not been initialized"

**Solution**: Ensure you're using the `deepspeed` launcher, not `torchrun`:

```bash
# ✓ Correct
deepspeed --num_gpus=2 main_finetune_qwen.py --deepspeed ...

# ✗ Incorrect
torchrun --nproc_per_node=2 main_finetune_qwen.py --deepspeed ...
```

### Issue: FusedAdam compilation errors

**Solution**: The implementation uses standard PyTorch AdamW optimizer (no compilation needed). If you still see FusedAdam errors, ensure you're using the latest code which creates the optimizer manually before DeepSpeed initialization.

**Note**: FusedAdam can provide ~5-10% speedup but requires CUDA compilation. Our implementation prioritizes compatibility over this marginal gain.

### Issue: Out of memory with DeepSpeed

**Solutions**:
1. Enable CPU offloading: `--deepspeed_config deepspeed_configs/zero2_offload_config.json`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Reduce batch size: `--batch_size 16`
4. Increase gradient accumulation: `--accum_iter 4`

### Issue: Slow training with CPU offloading

**Explanation**: CPU offloading trades memory for speed. This is expected.

**Solutions**:
- Use standard ZeRO-2 without offloading if you have enough GPU memory
- Increase `pin_memory` in DeepSpeed config for faster transfers
- Use faster CPU RAM (higher bandwidth)

### Issue: Checkpoint conversion fails

**Solution**: Ensure the checkpoint directory contains DeepSpeed files:

```bash
ls ./results/deepspeed_training/checkpoint-99/
# Should contain: *_optim_states.pt, *_model_states.pt, client_state.pt
```

## Advanced Configuration

### Custom DeepSpeed Config

Create a custom configuration file based on the templates:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_bucket_size": 5e8,  // Adjust for network bandwidth
    "reduce_bucket_size": 5e8
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0
}
```

### Environment Variables

Useful environment variables for debugging:

```bash
# Enable DeepSpeed logging
export DEEPSPEED_LOG_LEVEL=INFO

# Show memory usage
export CUDA_LAUNCH_BLOCKING=1

# Distributed debugging
export NCCL_DEBUG=INFO
```

## Performance Tips

1. **Batch size**: DeepSpeed allows 2-4x larger batch sizes. Increase from 8 to 32.
2. **Gradient checkpointing**: Combine with DeepSpeed for maximum memory savings.
3. **Mixed precision**: bfloat16 is enabled by default (better than fp16 for training).
4. **Communication**: Increase bucket sizes in config for faster multi-GPU communication.
5. **CPU offloading**: Use only if GPU memory is insufficient.

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review DeepSpeed logs: `cat ./results/deepspeed_training/deepspeed.log`
3. Compare with standard DDP training to isolate DeepSpeed-specific issues
4. Open an issue with logs and configuration details

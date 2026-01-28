#!/bin/bash
# ============================================================
# Qwen3-8B Linear Probe with Standard PyTorch DDP
# ============================================================
# This script performs linear probing using standard PyTorch DDP
#
# Prerequisites:
#   - Stage 1 checkpoint at: results/qwen_ddp/finetune/checkpoint-*
#     OR results/qwen_deepspeed/finetune/checkpoint-*
#
# Benefits of Standard DDP for Linear Probe:
#   - Simpler setup (no DeepSpeed config files)
#   - Fewer bugs (standard PyTorch, fewer edge cases)
#   - Better debugging (standard checkpoint format)
#   - Lower memory per GPU (18-22GB vs 25-30GB)
#   - Loads both DDP and DeepSpeed checkpoints
#
# When to use DeepSpeed instead:
#   - Larger batch sizes needed (256 vs 128)
#   - 4+ GPUs for maximum training speed
#   - Very large models (>8B parameters)
# ============================================================

set -e  # Exit on error

# Configuration
SEED=42
PRETRAINED_PATH="/workspace/cross_modality_llm/Qwen3-8B"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/qwen_deepspeed}"
DATASET="${DATASET:-cifar100}"
NB_CLASSES="${NB_CLASSES:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"  # Standard DDP batch size (vs 256 for DeepSpeed)
ACCUM_ITER="${ACCUM_ITER:-1}"
EPOCHS="${EPOCHS:-100}"

# Distributed training
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================================"
echo "Qwen3-8B Linear Probe with Standard PyTorch DDP"
echo "============================================================"

# Find latest checkpoint from fine-tuning
FINETUNE_OUTPUT="${OUTPUT_BASE}/finetune"

if [ ! -d "$FINETUNE_OUTPUT" ]; then
    echo "WARNING: Fine-tuning output directory not found: $FINETUNE_OUTPUT"
    echo "Checking alternative location: ./results/qwen_deepspeed/finetune"

    # Check if DeepSpeed checkpoint exists
    if [ -d "./results/qwen_deepspeed/finetune" ]; then
        FINETUNE_OUTPUT="./results/qwen_deepspeed/finetune"
        echo "✓ Found DeepSpeed checkpoint directory"
    else
        echo "ERROR: No fine-tuning checkpoint found"
        echo "Please run fine-tuning first using:"
        echo "  - qwen_deepspeed_experiment.sh (for DeepSpeed)"
        echo "  - or your custom fine-tuning script"
        exit 1
    fi
fi

# Auto-detect checkpoint format (directory = DeepSpeed, file = PyTorch)
LATEST_CHECKPOINT=$(find ${FINETUNE_OUTPUT} -maxdepth 1 \( -name "checkpoint-*.pth" -o -type d -name "checkpoint-*" \) | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in $FINETUNE_OUTPUT"
    echo "Expected format:"
    echo "  - PyTorch: checkpoint-*.pth (single file)"
    echo "  - DeepSpeed: checkpoint-*/ (directory)"
    exit 1
fi

# Set deepspeed_checkpoint flag if checkpoint is a directory (DeepSpeed format)
if [ -d "$LATEST_CHECKPOINT" ]; then
    DEEPSPEED_CHECKPOINT_FLAG="--deepspeed_checkpoint"
    echo "✓ Detected DeepSpeed checkpoint format: $LATEST_CHECKPOINT"

    # Verify DeepSpeed checkpoint structure
    if [ ! -f "$LATEST_CHECKPOINT/mp_rank_00_model_states.pt" ]; then
        echo "WARNING: DeepSpeed checkpoint may be corrupted"
        echo "Missing file: mp_rank_00_model_states.pt"
    fi
else
    DEEPSPEED_CHECKPOINT_FLAG=""
    echo "✓ Detected PyTorch checkpoint format: $LATEST_CHECKPOINT"
fi

LINPROBE_OUTPUT="${OUTPUT_BASE}/linear_probe"

echo ""
echo "Configuration:"
echo "  - Dataset: $DATASET ($NB_CLASSES classes)"
echo "  - Data Path: $DATA_PATH"
echo "  - Batch Size: $BATCH_SIZE (per GPU)"
echo "  - Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "  - Epochs: $EPOCHS"
echo "  - GPUs: $NUM_GPUS"
echo "  - Checkpoint: $LATEST_CHECKPOINT"
echo "  - Output: $LINPROBE_OUTPUT"
echo "  - Mode: Standard PyTorch DDP"
echo "============================================================"
echo ""
echo "Starting linear probing with DDP..."
echo ""

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    main_linprobe_qwen.py \
    $DEEPSPEED_CHECKPOINT_FLAG \
    --finetune_checkpoint $LATEST_CHECKPOINT \
    --pretrained_model_path $PRETRAINED_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --nb_classes $NB_CLASSES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --accum_iter $ACCUM_ITER \
    --input_size 224 \
    --blr 0.1 \
    --weight_decay 0 \
    --warmup_epochs 10 \
    --linear_probe \
    --dist_eval \
    --output_dir $LINPROBE_OUTPUT \
    --log_dir $LINPROBE_OUTPUT \
    --seed $SEED

echo ""
echo "============================================================"
echo "Linear Probing with DDP Completed!"
echo "============================================================"
echo ""
echo "Results Summary:"
echo "  - Checkpoint Used: $LATEST_CHECKPOINT"
echo "  - Output Directory: $LINPROBE_OUTPUT"
echo ""
echo "To view results:"
echo "  cat ${LINPROBE_OUTPUT}/log.txt"
echo "  cat ${LINPROBE_OUTPUT}/final_results.json"
echo ""
echo "Memory Usage Comparison:"
echo "  Standard DDP (batch_size=128): ~18-22GB per GPU (this run)"
echo "  DeepSpeed ZeRO-2 (batch_size=256): ~25-30GB per GPU"
echo ""
echo "Performance Comparison:"
echo "  Standard DDP: Baseline speed, simpler debugging"
echo "  DeepSpeed ZeRO-2: 1.3-1.5x faster on 4+ GPUs"
echo "============================================================"

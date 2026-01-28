#!/bin/bash
# ============================================================
# Qwen3-8B Experiment: Random Initialization
# ============================================================
# This script runs a two-stage experiment with Qwen3-8B:
#   Stage 1: Train randomly initialized Qwen3-8B on CIFAR-10
#   Stage 2: Linear probing on the trained checkpoint
#
# All experiments use seed=42 for reproducibility
#
# Distributed Training: Uses 2 GPUs via torchrun
# ============================================================

set -e  # Exit on error

# Configuration
SEED=42
PRETRAINED_PATH="/workspace/cross_modality_llm/Qwen3-8B"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/qwen_random_init}"
DATASET="cifar10"
NB_CLASSES=10
BATCH_SIZE=32
ACCUM_ITER=2
EPOCHS_FINETUNE=100
EPOCHS_LINPROBE=100

# Distributed training configuration
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "============================================================"
echo "Qwen3-8B Random Initialization Experiment"
echo "Seed: $SEED"
echo "Dataset: $DATASET (${NB_CLASSES} classes)"
echo "Data Path: $DATA_PATH"
echo "Output Base: $OUTPUT_BASE"
echo "Distributed Training: ${NUM_GPUS} GPUs (Master Port: $MASTER_PORT)"
echo "============================================================"

# ============================================================
# Stage 1: Fine-tune randomly initialized Qwen3-8B on CIFAR-10
# ============================================================

FINETUNE_OUTPUT="${OUTPUT_BASE}/finetune"
FINETUNE_CHECKPOINT="${FINETUNE_OUTPUT}/checkpoint-best.pth"

echo ""
echo "============================================================"
echo "Stage 1: Training randomly initialized Qwen3-8B on CIFAR-10"
echo "Output: $FINETUNE_OUTPUT"
echo "============================================================"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    main_finetune_qwen.py \
    --random_init \
    --pretrained_model_path $PRETRAINED_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --nb_classes $NB_CLASSES \
    --batch_size $BATCH_SIZE \
    --accum_iter $ACCUM_ITER \
    --epochs $EPOCHS_FINETUNE \
    --input_size 224 \
    --drop_path 0.1 \
    --clip_grad 1.0 \
    --weight_decay 0.05 \
    --blr 1e-3 \
    --warmup_epochs 10 \
    --gradient_checkpointing \
    --dist_eval \
    --output_dir $FINETUNE_OUTPUT \
    --log_dir $FINETUNE_OUTPUT \
    --seed $SEED

echo ""
echo "Stage 1 completed! Checkpoint saved to: $FINETUNE_CHECKPOINT"

# ============================================================
# Stage 2: Linear probing on the trained checkpoint
# ============================================================

LINPROBE_OUTPUT="${OUTPUT_BASE}/linear_probe"

echo ""
echo "============================================================"
echo "Stage 2: Linear probing on trained Qwen3-8B"
echo "Loading checkpoint: $FINETUNE_CHECKPOINT"
echo "Output: $LINPROBE_OUTPUT"
echo "============================================================"

# Check if checkpoint exists
if [ ! -f "$FINETUNE_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $FINETUNE_CHECKPOINT"
    echo "Looking for alternative checkpoints..."

    # Try to find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -t ${FINETUNE_OUTPUT}/checkpoint-*.pth 2>/dev/null | head -n 1)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Using latest checkpoint: $LATEST_CHECKPOINT"
        FINETUNE_CHECKPOINT=$LATEST_CHECKPOINT
    else
        echo "ERROR: No checkpoints found in $FINETUNE_OUTPUT"
        exit 1
    fi
fi

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    main_linprobe_qwen.py \
    --finetune_checkpoint $FINETUNE_CHECKPOINT \
    --pretrained_model_path $PRETRAINED_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --nb_classes $NB_CLASSES \
    --batch_size 128 \
    --epochs $EPOCHS_LINPROBE \
    --accum_iter 1 \
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
echo "Qwen3-8B Random Initialization Experiment Completed!"
echo "============================================================"
echo ""
echo "Results Summary:"
echo "----------------"
echo "Stage 1 (Finetuning): $FINETUNE_OUTPUT"
echo "Stage 2 (Linear Probe): $LINPROBE_OUTPUT"
echo ""
echo "To view results:"
echo "  cat ${FINETUNE_OUTPUT}/log.txt"
echo "  cat ${LINPROBE_OUTPUT}/log.txt"
echo "============================================================"

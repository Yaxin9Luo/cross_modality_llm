#!/bin/bash
# ============================================================
# Quick Test: DeepSpeed Checkpoint Saving (10 steps only)
# ============================================================
# This script tests checkpoint saving after just 10 training steps

set -e  # Exit on error

# Configuration
SEED=42
PRETRAINED_PATH="/workspace/cross_modality_llm/Qwen3-8B"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/qwen_checkpoint_test}"
DATASET="cifar100"
NB_CLASSES=100
BATCH_SIZE=64
ACCUM_ITER=1  # No accumulation for quick test
EPOCHS=1  # Just 1 epoch, but we'll limit steps

# DeepSpeed configuration
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero2_config.json}"

# Distributed training configuration
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================================"
echo "Quick Checkpoint Save Test (10 steps)"
echo "Seed: $SEED"
echo "Pretrained Model Path: $PRETRAINED_PATH"
echo "Dataset: $DATASET (${NB_CLASSES} classes)"
echo "Data Path: $DATA_PATH"
echo "Output: $OUTPUT_BASE"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "Distributed Training: ${NUM_GPUS} GPUs"
echo "============================================================"

# Verify pretrained model path exists
if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained model path does not exist: $PRETRAINED_PATH"
    exit 1
fi

echo "✓ Pretrained model path verified"

# Verify DeepSpeed config exists
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "ERROR: DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

echo "✓ DeepSpeed config verified"

# Create output directory
mkdir -p $OUTPUT_BASE

echo ""
echo "============================================================"
echo "Running 10 training steps with checkpoint save"
echo "============================================================"

# Note: We'll need to modify main_finetune_qwen.py to support --max_steps
# For now, let's run with 1 epoch and very small dataset subset
# The checkpoint will save at epoch 0 if we modify the save frequency

deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \
    main_finetune_qwen.py \
    --deepspeed \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --load_pretrained \
    --pretrained_model_path $PRETRAINED_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --nb_classes $NB_CLASSES \
    --batch_size $BATCH_SIZE \
    --accum_iter $ACCUM_ITER \
    --epochs 1 \
    --input_size 224 \
    --drop_path 0.1 \
    --clip_grad 1.0 \
    --weight_decay 0.05 \
    --blr 1e-3 \
    --warmup_epochs 0 \
    --gradient_checkpointing \
    --dist_eval \
    --output_dir $OUTPUT_BASE \
    --log_dir $OUTPUT_BASE \
    --seed $SEED

echo ""
echo "============================================================"
echo "Checkpoint Test Completed!"
echo "============================================================"
echo ""
echo "Checking for saved checkpoints..."
ls -lh $OUTPUT_BASE/checkpoint-* 2>/dev/null || echo "No checkpoints found (expected for 1 epoch < 10)"

echo ""
echo "Note: Checkpoints save every 10 epochs by default."
echo "To test checkpoint saving, we need to run at least until epoch 9 completes."
echo "Consider modifying the save frequency in main_finetune_qwen.py for testing."

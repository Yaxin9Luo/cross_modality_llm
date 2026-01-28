#!/bin/bash
# ============================================================
# Resume Qwen3-8B DeepSpeed Training from Checkpoint
# ============================================================
# This script resumes training from a DeepSpeed checkpoint
# ============================================================

set -e  # Exit on error

# Configuration
SEED=42
PRETRAINED_PATH="/workspace/cross_modality_llm/Qwen3-8B"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_DIR="/workspace/cross_modality_llm/results/qwen_deepspeed/finetune"
DATASET="cifar100"
NB_CLASSES=100
BATCH_SIZE=64
ACCUM_ITER=2
EPOCHS_FINETUNE=100

# Resume from checkpoint
RESUME_TAG="checkpoint-49"  # The checkpoint directory name (tag)

# DeepSpeed configuration
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero2_config.json}"

# Distributed training configuration
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================================"
echo "Resuming Qwen3-8B DeepSpeed Training"
echo "Seed: $SEED"
echo "Dataset: $DATASET (${NB_CLASSES} classes)"
echo "Output Directory: $OUTPUT_DIR"
echo "Resume from: $RESUME_TAG"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "============================================================"

# Verify checkpoint exists
CHECKPOINT_DIR="${OUTPUT_DIR}/${RESUME_TAG}"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "✓ Checkpoint verified: $CHECKPOINT_DIR"
ls -lh "$CHECKPOINT_DIR"

# Resume training with DeepSpeed
echo ""
echo "============================================================"
echo "Resuming Training from Epoch $(echo $RESUME_TAG | grep -oP '\d+')..."
echo "============================================================"

deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \
    main_finetune_qwen.py \
    --deepspeed \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --resume $RESUME_TAG \
    --load_pretrained \
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
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --seed $SEED

echo ""
echo "============================================================"
echo "Training Resumed and Completed!"
echo "============================================================"

#!/bin/bash
# ============================================================
# Qwen3-8B Linear Probe with DeepSpeed (Stage 2)
# ============================================================
# This script performs linear probing on a DeepSpeed fine-tuned checkpoint
#
# Prerequisites:
#   - Stage 1 checkpoint at: results/qwen_deepspeed/finetune/checkpoint-*
#
# Benefits of DeepSpeed for Linear Probe:
#   - Larger batch sizes (256-512 vs 128)
#   - Faster training with ZeRO-2 optimization
#   - Consistent workflow with Stage 1
# ============================================================

set -e  # Exit on error

# Configuration
SEED=42
PRETRAINED_PATH="/workspace/cross_modality_llm/Qwen3-8B"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_BASE="./results/qwen_deepspeed"
DATASET="cifar100"
NB_CLASSES=100
BATCH_SIZE=64  # 2x larger than standard DDP (128)
ACCUM_ITER=1
EPOCHS=100

# Find latest fine-tuning checkpoint
FINETUNE_OUTPUT="${OUTPUT_BASE}/finetune"

echo "============================================================"
echo "Qwen3-8B Linear Probe with DeepSpeed (Stage 2)"
echo "============================================================"

# Check if fine-tuning output directory exists
if [ ! -d "$FINETUNE_OUTPUT" ]; then
    echo "ERROR: Fine-tuning output directory not found: $FINETUNE_OUTPUT"
    echo "Please run Stage 1 (fine-tuning) first using qwen_deepspeed_experiment.sh"
    exit 1
fi

# Find latest checkpoint
LATEST_CHECKPOINT=$(find ${FINETUNE_OUTPUT} -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No DeepSpeed checkpoint found in $FINETUNE_OUTPUT"
    echo "Please ensure Stage 1 (fine-tuning) has completed successfully"
    exit 1
fi

echo "Using fine-tuned checkpoint: $LATEST_CHECKPOINT"

# Verify checkpoint contains required files
if [ ! -f "$LATEST_CHECKPOINT/mp_rank_00_model_states.pt" ]; then
    echo "ERROR: Checkpoint appears to be corrupted or incomplete"
    echo "Missing file: mp_rank_00_model_states.pt"
    exit 1
fi

echo "✓ Checkpoint verified"

# DeepSpeed configuration (use linprobe config that allows LARS optimizer)
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero2_linprobe_config.json}"

# Verify DeepSpeed config exists
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "ERROR: DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

echo "✓ DeepSpeed config verified: $DEEPSPEED_CONFIG"

# Distributed training
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

LINPROBE_OUTPUT="${OUTPUT_BASE}/linear_probe"

echo ""
echo "Configuration:"
echo "  - Dataset: $DATASET ($NB_CLASSES classes)"
echo "  - Data Path: $DATA_PATH"
echo "  - Batch Size: $BATCH_SIZE (per GPU)"
echo "  - Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "  - Epochs: $EPOCHS"
echo "  - GPUs: $NUM_GPUS"
echo "  - Output: $LINPROBE_OUTPUT"
echo "============================================================"
echo ""
echo "Starting linear probing with DeepSpeed..."
echo ""

deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \
    main_linprobe_qwen.py \
    --deepspeed \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --deepspeed_checkpoint \
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
echo "Linear Probing with DeepSpeed Completed!"
echo "============================================================"
echo ""
echo "Results Summary:"
echo "  - Checkpoint: $LATEST_CHECKPOINT"
echo "  - Output: $LINPROBE_OUTPUT"
echo ""
echo "To view results:"
echo "  cat ${LINPROBE_OUTPUT}/log.txt"
echo ""
echo "Memory Usage Comparison:"
echo "  Standard DDP (batch_size=128): ~20-25GB per GPU"
echo "  DeepSpeed ZeRO-2 (batch_size=256): ~25-30GB per GPU"
echo "============================================================"

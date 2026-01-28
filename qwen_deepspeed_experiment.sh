#!/bin/bash
# ============================================================
# Qwen3-8B Experiment: DeepSpeed ZeRO-2 Training
# ============================================================
# This script demonstrates DeepSpeed ZeRO-2 training for Qwen3-8B:
#   - Stage 1: Fine-tune pretrained Qwen3-8B with DeepSpeed ZeRO-2
#   - Stage 2: Linear probing (optional, can use standard DDP)
#
# Benefits of DeepSpeed ZeRO-2:
#   - 50-60% GPU memory reduction
#   - Enables larger batch sizes
#   - CPU offloading option for extreme memory savings
#
# Distributed Training: Uses DeepSpeed launcher
# ============================================================

set -e  # Exit on error

# Configuration
SEED=42
PRETRAINED_PATH="/workspace/cross_modality_llm/Qwen3-14B"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/qwen_deepspeed_14b_random_init}"
DATASET="cifar100"
NB_CLASSES=100
BATCH_SIZE=32  # Can use larger batch size with ZeRO-2 (vs 8 with DDP)
ACCUM_ITER=2
EPOCHS_FINETUNE=100
EPOCHS_LINPROBE=100

# DeepSpeed configuration
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero2_config.json}"
USE_OFFLOAD="${USE_OFFLOAD:-false}"  # Set to true for CPU offloading

# Distributed training configuration
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================================"
echo "Qwen3-14B DeepSpeed ZeRO-2 Experiment with Random Initialization"
echo "Seed: $SEED"
echo "Pretrained Model Path: $PRETRAINED_PATH"
echo "Dataset: $DATASET (${NB_CLASSES} classes)"
echo "Data Path: $DATA_PATH"
echo "Output Base: $OUTPUT_BASE"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "CPU Offload: $USE_OFFLOAD"
echo "Distributed Training: ${NUM_GPUS} GPUs (Master Port: $MASTER_PORT)"
echo "============================================================"

# Use offload config if requested
if [ "$USE_OFFLOAD" = "true" ]; then
    DEEPSPEED_CONFIG="deepspeed_configs/zero2_offload_config.json"
    echo "Using CPU offload configuration for maximum memory savings"
fi

# Verify pretrained model path exists
if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained model path does not exist: $PRETRAINED_PATH"
    echo "Please ensure the Qwen3-14B model files are available at this location."
    exit 1
fi

echo "✓ Pretrained model path verified"

# Verify DeepSpeed config exists
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "ERROR: DeepSpeed config not found: $DEEPSPEED_CONFIG"
    echo "Please ensure the DeepSpeed configuration file exists."
    exit 1
fi

echo "✓ DeepSpeed config verified"

# ============================================================
# Stage 1: Fine-tune with DeepSpeed ZeRO-2
# ============================================================

FINETUNE_OUTPUT="${OUTPUT_BASE}/finetune"

echo ""
echo "============================================================"
echo "Stage 1: Fine-tuning with DeepSpeed ZeRO-2 and Random Initialization"
echo "Loading pretrained weights from: $PRETRAINED_PATH"
echo "Output: $FINETUNE_OUTPUT"
echo "Expected memory savings: 50-60% vs standard DDP"
echo "============================================================"

deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \
    main_finetune_qwen.py \
    --deepspeed \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --random_init \
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
echo "Stage 1 completed! DeepSpeed checkpoint saved to: $FINETUNE_OUTPUT"

# ============================================================
# Stage 2: Linear probing with DeepSpeed
# ============================================================

LINPROBE_OUTPUT="${OUTPUT_BASE}/linear_probe"

echo ""
echo "============================================================"
echo "Stage 2: Linear probing with DeepSpeed and Random Initialization"
echo "Note: Using DeepSpeed for larger batch sizes and consistency"
echo "Output: $LINPROBE_OUTPUT"
echo "============================================================"

# Find the latest DeepSpeed checkpoint
LATEST_CHECKPOINT_DIR=$(find ${FINETUNE_OUTPUT} -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT_DIR" ]; then
    echo "WARNING: No DeepSpeed checkpoint found in $FINETUNE_OUTPUT"
    echo "Skipping linear probing stage."
else
    echo "Using DeepSpeed checkpoint: $LATEST_CHECKPOINT_DIR"

    # Verify checkpoint integrity
    if [ ! -f "$LATEST_CHECKPOINT_DIR/mp_rank_00_model_states.pt" ]; then
        echo "ERROR: Checkpoint appears to be corrupted or incomplete"
        echo "Missing file: mp_rank_00_model_states.pt"
        echo "Skipping linear probing stage."
    else
        echo "✓ Checkpoint verified"

        # Use linprobe-specific config that allows LARS optimizer
        LINPROBE_DEEPSPEED_CONFIG="deepspeed_configs/zero2_linprobe_config.json"

        deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \
            main_linprobe_qwen.py \
            --deepspeed \
            --deepspeed_config $LINPROBE_DEEPSPEED_CONFIG \
            --deepspeed_checkpoint \
            --finetune_checkpoint $LATEST_CHECKPOINT_DIR \
            --pretrained_model_path $PRETRAINED_PATH \
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --nb_classes $NB_CLASSES \
            --batch_size 256 \
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
        echo "Stage 2 completed! Results saved to: $LINPROBE_OUTPUT"
    fi
fi

echo ""
echo "============================================================"
echo "Qwen3-8B DeepSpeed ZeRO-2 Experiment with Random Initialization Completed!"
echo "============================================================"
echo ""
echo "Results Summary:"
echo "----------------"
echo "Pretrained Model: $PRETRAINED_PATH"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "Stage 1 (DeepSpeed Finetuning): $FINETUNE_OUTPUT"
echo "Stage 2 (DeepSpeed Linear Probe): $LINPROBE_OUTPUT"
echo ""
echo "To view results:"
echo "  Stage 1: cat ${FINETUNE_OUTPUT}/log.txt"
echo "  Stage 2: cat ${LINPROBE_OUTPUT}/log.txt"
echo ""
echo "Memory Usage Comparison:"
echo "  Standard DDP Fine-tuning (batch_size=8): ~40-50GB per GPU"
echo "  DeepSpeed ZeRO-2 Fine-tuning (batch_size=64): ~20-25GB per GPU"
echo "  DeepSpeed ZeRO-2 + Offload (batch_size=64): ~15-20GB per GPU"
echo "  DeepSpeed ZeRO-2 Linear Probe (batch_size=256): ~25-30GB per GPU"
echo ""
echo "To convert DeepSpeed checkpoint to standard PyTorch format:"
echo "  python convert_deepspeed_checkpoint.py --checkpoint $FINETUNE_OUTPUT/checkpoint-* --output converted_checkpoint.pth"
echo ""
echo "To run only Stage 2 (linear probe) separately:"
echo "  ./qwen_deepspeed_linprobe.sh"
echo "============================================================"

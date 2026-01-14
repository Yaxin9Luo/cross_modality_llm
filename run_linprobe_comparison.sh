#!/bin/bash
# ============================================================
# Linear Probing Comparison: LBBT (GPT-2) vs DINO
# ============================================================
# This script runs linear probing evaluation for comparing:
#   - LBBT: Language-Based Backbone Transfer (GPT-2 Medium: 355M params)
#   - DINO: Self-DIstillation with NO labels (standard vision SSL)
#
# For FAIR comparison (similar parameter count):
#   - LBBT:          GPT-2 Medium  = 355M params, 1024 dim, 24 layers
#   - DINOv2 ViT-L:  ViT-L/14      = 304M params, 1024 dim, 24 layers
# ============================================================

# Configuration
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/data}"
OUTPUT_BASE="${OUTPUT_BASE:-./linprobe_comparison_results}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# LBBT checkpoint (optional - leave empty for language-pretrained GPT-2)
LBBT_CHECKPOINT="${LBBT_CHECKPOINT:-}"

# Dataset selection: cifar10, cifar100, imagenet100, tiny-imagenet
DATASET="${DATASET:-cifar10}"

echo "============================================================"
echo "Linear Probing Comparison: LBBT vs DINO"
echo "Dataset: $DATASET"
echo "Data Path: $DATA_PATH"
echo "Output: $OUTPUT_BASE"
echo "============================================================"

# ============================================================
# DINO Models Comparison
# ============================================================

run_dino() {
    local model_name=$1
    echo ""
    echo "============================================================"
    echo "Running DINO: $model_name on $DATASET"
    echo "============================================================"
    
    python linear_probe_comparison.py \
        --method dino \
        --dino_model $model_name \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --num_workers $NUM_WORKERS \
        --output_dir $OUTPUT_BASE \
        --blr 0.1 \
        --warmup_epochs 10
}

# ============================================================
# LBBT (GPT-2) Evaluation
# ============================================================

run_lbbt() {
    echo ""
    echo "============================================================"
    echo "Running LBBT (GPT-2) on $DATASET"
    echo "============================================================"
    
    local extra_args=""
    if [ -n "$LBBT_CHECKPOINT" ]; then
        extra_args="--lbbt_checkpoint $LBBT_CHECKPOINT"
    else
        extra_args="--lbbt_pretrained"
    fi
    
    python linear_probe_comparison.py \
        --method lbbt \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --num_workers $NUM_WORKERS \
        --output_dir $OUTPUT_BASE \
        --blr 0.1 \
        --warmup_epochs 10 \
        $extra_args
}

# ============================================================
# Run Experiments
# ============================================================

# Parse command line arguments
RUN_ALL=false
RUN_FAIR=false
RUN_LBBT=false
RUN_DINO_S=false
RUN_DINO_B=false
RUN_DINOV2_S=false
RUN_DINOV2_B=false
RUN_DINOV2_L=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --fair)
            # Fair comparison: LBBT vs DINOv2-L (similar params)
            RUN_FAIR=true
            shift
            ;;
        --lbbt)
            RUN_LBBT=true
            shift
            ;;
        --dino-s)
            RUN_DINO_S=true
            shift
            ;;
        --dino-b)
            RUN_DINO_B=true
            shift
            ;;
        --dinov2-s)
            RUN_DINOV2_S=true
            shift
            ;;
        --dinov2-b)
            RUN_DINOV2_B=true
            shift
            ;;
        --dinov2-l)
            RUN_DINOV2_L=true
            shift
            ;;
        --dataset)
            DATASET=$2
            shift 2
            ;;
        --data-path)
            DATA_PATH=$2
            shift 2
            ;;
        --lbbt-checkpoint)
            LBBT_CHECKPOINT=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fair             ⭐ RECOMMENDED: Fair comparison (LBBT vs DINOv2 ViT-L/14)"
            echo "  --all              Run all models"
            echo "  --lbbt             Run LBBT (GPT-2 Medium, 355M) only"
            echo "  --dinov2-l         Run DINOv2 ViT-L/14 (304M) only"
            echo "  --dinov2-b         Run DINOv2 ViT-B/14 (86M) only"
            echo "  --dinov2-s         Run DINOv2 ViT-S/14 (22M) only"
            echo "  --dino-b           Run DINO ViT-B/16 (85M) only"
            echo "  --dino-s           Run DINO ViT-S/16 (21M) only"
            echo "  --dataset NAME     Dataset to use (cifar10, cifar100, imagenet100, tiny-imagenet)"
            echo "  --data-path PATH   Path to dataset"
            echo "  --lbbt-checkpoint  Path to LBBT checkpoint (optional)"
            echo ""
            echo "Model Comparison (Parameter Count):"
            echo "  LBBT (GPT-2 Medium) : 355M params, 1024 dim, 24 layers"
            echo "  DINOv2 ViT-L/14     : 304M params, 1024 dim, 24 layers  ⭐ Fair comparison"
            echo "  DINOv2 ViT-B/14     : 86M params,  768 dim,  12 layers"
            echo ""
            echo "Example (fair comparison on CIFAR-10):"
            echo "  $0 --fair --dataset cifar10"
            echo ""
            echo "Environment variables:"
            echo "  DATA_PATH          Dataset path (default: /root/autodl-tmp/data)"
            echo "  OUTPUT_BASE        Output directory (default: ./linprobe_comparison_results)"
            echo "  BATCH_SIZE         Batch size (default: 128)"
            echo "  EPOCHS             Number of epochs (default: 100)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default: run fair comparison if no specific model selected
if ! $RUN_LBBT && ! $RUN_DINO_S && ! $RUN_DINO_B && ! $RUN_DINOV2_S && ! $RUN_DINOV2_B && ! $RUN_DINOV2_L && ! $RUN_ALL; then
    RUN_FAIR=true
fi

# Execute experiments

# Fair comparison (recommended): LBBT vs DINOv2 ViT-L/14
if $RUN_FAIR; then
    echo ""
    echo "⭐ Running FAIR COMPARISON: LBBT (355M) vs DINOv2 ViT-L/14 (304M)"
    echo ""
    run_lbbt
    run_dino "dinov2_vitl14"
fi

if $RUN_ALL || $RUN_LBBT; then
    run_lbbt
fi

if $RUN_ALL || $RUN_DINOV2_L; then
    run_dino "dinov2_vitl14"
fi

if $RUN_ALL || $RUN_DINO_S; then
    run_dino "dino_vits16"
fi

if $RUN_ALL || $RUN_DINO_B; then
    run_dino "dino_vitb16"
fi

if $RUN_ALL || $RUN_DINOV2_S; then
    run_dino "dinov2_vits14"
fi

if $RUN_ALL || $RUN_DINOV2_B; then
    run_dino "dinov2_vitb14"
fi

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_BASE"
echo "============================================================"

# ============================================================
# Aggregate Results
# ============================================================

echo ""
echo "Results Summary:"
echo "----------------"

for dir in $OUTPUT_BASE/*/; do
    if [ -f "$dir/final_results.json" ]; then
        method=$(basename "$dir" | cut -d'_' -f1)
        dataset=$(basename "$dir" | cut -d'_' -f2)
        model=$(basename "$dir" | cut -d'_' -f3-)
        acc=$(cat "$dir/final_results.json" | python -c "import sys, json; print(f\"{json.load(sys.stdin)['best_acc1']:.2f}%\")" 2>/dev/null || echo "N/A")
        printf "%-10s %-15s %-20s: %s\n" "$method" "$dataset" "$model" "$acc"
    fi
done

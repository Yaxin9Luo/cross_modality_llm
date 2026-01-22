# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project comparing **cross-modality transfer learning** approaches for vision tasks:
- **LBBT (Language-Based Backbone Transfer)**: Adapting language models (GPT-2, Qwen) for vision tasks
- **DINO/DINOv2**: Standard self-supervised vision learning baselines

The core research question: Can pretrained language model representations transfer to vision tasks through patch embeddings?

## Environment Setup

### Installation
```bash
# Using uv (recommended)
uv sync
uv sync --extra wandb  # with W&B support

# Or using pip
pip install -r requirements.txt

# Setup script (includes timm patching for PyTorch 2.x)
./setup.sh
```

### Critical Dependencies
- **timm==0.3.2** (STRICT version requirement - enforced by MAE codebase)
- PyTorch >= 2.4.0
- transformers >= 4.44.0
- Python >= 3.10

### Code Quality Tools
```bash
# Formatting
black . --line-length 100

# Linting
ruff check .

# Type checking (optional - ignores missing imports)
mypy .
```

## Architecture Overview

### Model Architectures

The project implements three main model families:

#### 1. LBBT with GPT-2 ([model_llm.py](model_llm.py))
- `MAE_GPT2_Classifier`: GPT-2 Medium (355M params, 1024-dim, 24 layers)
- Architecture: `Image → PatchEmbed(16x16) → GPT-2 → Classifier`
- Key feature: Uses `inputs_embeds` to bypass tokenization
- Classification: Uses last token representation (GPT-style pooling)
- Supports: pretrained language weights, linear probing, layer freezing

#### 2. LBBT with Qwen ([model_qwen.py](model_qwen.py))
- `MAE_Qwen3_Classifier`: Qwen3-8B (8B params, 4096-dim)
- Architecture: `Image → PatchEmbed(16x16) → Qwen3-8B → Classifier`
- Uses bfloat16 for memory efficiency
- Supports: gradient checkpointing, pretrained weights, linear probing
- Model path: `ckpts/Qwen3-8B` (local) or `Qwen/Qwen2.5-8B` (HuggingFace)

#### 3. DINO Baselines ([models_dino.py](models_dino.py))
- `DINOClassifier`: Original DINO (ViT-S/B/16/8, ResNet50)
- `DINOv2Classifier`: DINOv2 (ViT-S/B/L/14)
- Fair comparison: DINOv2-ViT-L/14 (304M params) ≈ GPT-2 Medium (355M params)
- Uses BatchNorm + Linear classifier (MoCo v3 protocol)
- Loads pretrained weights from `torch.hub` (facebookresearch/dino)

### Cross-Modality Transfer Mechanism

The key innovation is adapting language models for vision:

```
Text Domain (Pretraining):          Vision Domain (Transfer):
[Token Embeddings]                  [Patch Embeddings]
        ↓                                   ↓
  GPT-2 Layers                        GPT-2 Layers (frozen/finetuned)
        ↓                                   ↓
[Language Head]                      [Classification Head]
```

**Critical insight**: Language model representations can transfer to vision tasks when:
1. Images are converted to patch sequences (16x16 patches)
2. Patches are embedded to match LM hidden dimension (1024 for GPT-2 Medium)
3. LM processes patches via `inputs_embeds` (bypassing tokenization)

### Training Modes

1. **Full Finetuning**: Train all parameters (patch_embed + backbone + classifier)
2. **Linear Probing**: Freeze backbone, train only classifier + patch_embed
3. **Layer-wise Freezing**: Freeze specific layers (e.g., first N layers)

### Supported Datasets

Configured via `--dataset` and `--nb_classes`:
- CIFAR-10 (10 classes)
- CIFAR-100 (100 classes)
- Tiny-ImageNet (200 classes)
- ImageNet-100 (100 classes)
- ImageNet (1000 classes)

Data loading via `util/datasets.py` expects ImageFolder structure: `{train,val}/<class>/*.jpg`

## Common Commands

### Training

#### Full Finetuning (GPT-2)
```bash
python main_finetune.py \
    --batch_size 64 \
    --epochs 200 \
    --input_size 224 \
    --drop_path 0.1 \
    --clip_grad 1.0 \
    --weight_decay 0.05 \
    --data_path /path/to/tiny-imagenet-200 \
    --nb_classes 200 \
    --dataset tiny-imagenet \
    --output_dir ./results/finetune \
    --log_dir ./results/log
```

#### Linear Probing (GPT-2)
```bash
python main_finetune.py \
    --batch_size 128 \
    --epochs 200 \
    --linear_probe \
    --gpt2_checkpoint /path/to/checkpoint.pth \
    --data_path /path/to/dataset \
    --nb_classes 200 \
    --dataset tiny-imagenet \
    --output_dir ./results/linear_probe
```

#### Finetuning with Qwen
```bash
python main_finetune_qwen.py \
    --batch_size 32 \
    --epochs 100 \
    --gradient_checkpointing \
    --data_path /path/to/dataset \
    --nb_classes 10 \
    --dataset cifar10 \
    --output_dir ./results/qwen_finetune
```

### Evaluation & Comparison

#### Fair Comparison: LBBT vs DINOv2 (Recommended)
```bash
# Run both LBBT (GPT-2 Medium, 355M) and DINOv2 ViT-L/14 (304M)
./run_linprobe_comparison.sh --fair --dataset cifar10

# Custom dataset
DATASET=cifar100 DATA_PATH=/path/to/data ./run_linprobe_comparison.sh --fair
```

#### Run Specific Models
```bash
# LBBT only (with language-pretrained GPT-2)
./run_linprobe_comparison.sh --lbbt --dataset cifar10

# LBBT with custom checkpoint
LBBT_CHECKPOINT=/path/to/checkpoint.pth ./run_linprobe_comparison.sh --lbbt

# DINOv2 ViT-L/14 only
./run_linprobe_comparison.sh --dinov2-l --dataset cifar10

# All models
./run_linprobe_comparison.sh --all --dataset cifar10
```

#### Direct Python Evaluation
```bash
# LBBT (GPT-2) linear probe
python linear_probe_comparison.py \
    --method lbbt \
    --lbbt_pretrained \
    --dataset cifar10 \
    --data_path /path/to/data \
    --batch_size 128 \
    --epochs 100 \
    --output_dir ./results

# DINOv2 ViT-L/14 linear probe
python linear_probe_comparison.py \
    --method dino \
    --dino_model dinov2_vitl14 \
    --dataset cifar10 \
    --data_path /path/to/data \
    --batch_size 128 \
    --epochs 100 \
    --output_dir ./results
```

### Analysis & Visualization

```bash
# Analyze model weights
python analyze_weights.py --checkpoint /path/to/checkpoint.pth

# Layer-wise weight analysis
python analyze_weights_by_layer.py --checkpoint /path/to/checkpoint.pth

# Activation visualization
python activation_visualize.py --checkpoint /path/to/checkpoint.pth

# Gradient visualization
python gradient_visualize.py --checkpoint /path/to/checkpoint.pth

# Feature space analysis (t-SNE, PCA)
python feature_space_analysis.py --checkpoint /path/to/checkpoint.pth

# Neuron activation heatmaps
python neuron_activation_heatmap.py --checkpoint /path/to/checkpoint.pth

# Loss landscape visualization
python plot_loss_landscape.py --checkpoint /path/to/checkpoint.pth
```

### Checkpoint Management

```bash
# Save model weights
python save_weights.py --checkpoint /path/to/checkpoint.pth --output /path/to/weights.pt

# Convert to HuggingFace format
python convert_to_hf_format.py --checkpoint /path/to/checkpoint.pth --output ./hf_model
```

## Key Implementation Details

### Patch Embedding
- All models use 16x16 patches (consistent with ViT)
- Input: `[B, 3, 224, 224]` → Output: `[B, 196, hidden_dim]`
- Initialized with Xavier uniform

### Classification Head
- LBBT: Single Linear layer, initialized with normal(std=0.02)
- DINO: BatchNorm (affine=False) + Linear, initialized with trunc_normal(std=0.01)

### Learning Rate Scaling
- Base LR scaling: `lr = blr * batch_size / 256`
- Typical `blr` values:
  - Full finetuning: 1e-3
  - Linear probing: 0.1
- Layer-wise LR decay: 0.75 (for full finetuning)

### Optimizer Configuration
- Full finetuning: AdamW with weight decay 0.05
- Linear probing: LARS or SGD with weight decay 0 (MoCo v1 protocol)

### Data Augmentation
- Training: AutoAugment (rand-m9-mstd0.5-inc1), RandomResizedCrop, RandomHorizontalFlip
- Evaluation: Resize → CenterCrop → Normalize
- Normalization: ImageNet mean/std

### bfloat16 Training (Qwen)
- Qwen models use bfloat16 for memory efficiency
- Requires `NoOpScaler` instead of `NativeScaler` (no gradient scaling needed)
- Forward pass: patch_embed output converted to bfloat16

## Directory Structure

```
/workspace/cross_modality_llm/
├── model_llm.py              # LBBT GPT-2 implementation
├── model_qwen.py             # LBBT Qwen implementation
├── models_dino.py            # DINO/DINOv2 baselines
├── main_finetune.py          # Training script (GPT-2)
├── main_finetune_qwen.py     # Training script (Qwen)
├── main_linprobe.py          # Linear probing (ViT baseline)
├── main_linprobe_qwen.py     # Linear probing (Qwen)
├── linear_probe_comparison.py # Unified comparison script
├── engine_finetune.py        # Training/eval loops
├── engine_pretrain.py        # Pretraining loops
├── util/
│   ├── datasets.py           # Dataset loading & transforms
│   ├── misc.py               # Training utilities, NativeScaler
│   ├── lars.py               # LARS optimizer
│   ├── lr_decay.py           # Layer-wise LR decay
│   ├── lr_sched.py           # LR schedulers
│   ├── pos_embed.py          # Position embedding utilities
│   └── crop.py               # RandomResizedCrop
├── *.sh                      # Training/evaluation scripts
└── *_visualize.py, *_analysis.py  # Analysis tools
```

## Results Storage

Training outputs are saved to:
- `--output_dir`: Model checkpoints (`checkpoint-{epoch}.pth`, `checkpoint-best.pth`)
- `--log_dir`: TensorBoard logs
- Results include: `log.txt`, `args.json`, `final_results.json`

Linear probe comparison results:
- Default: `./linprobe_comparison_results/{method}_{dataset}_{model}/`
- Contains: checkpoints, logs, `final_results.json`

## Parameter Counts (for Fair Comparison)

| Model | Parameters | Hidden Dim | Layers |
|-------|-----------|------------|--------|
| GPT-2 Medium (LBBT) | 355M | 1024 | 24 |
| DINOv2 ViT-L/14 | 304M | 1024 | 24 |
| DINOv2 ViT-B/14 | 86M | 768 | 12 |
| Qwen3-8B (LBBT) | 8B | 4096 | - |

**Recommended fair comparison**: LBBT (GPT-2 Medium) vs DINOv2 ViT-L/14

## Common Issues

### timm Version
If you see import errors or compatibility issues, verify timm version:
```bash
python -c "import timm; print(timm.__version__)"  # Must be 0.3.2
```

### CUDA Out of Memory (Qwen)
- Enable gradient checkpointing: `--gradient_checkpointing`
- Reduce batch size: `--batch_size 16`
- Use gradient accumulation: `--accum_iter 4`

### Checkpoint Loading
- GPT-2 checkpoints contain: `model`, `optimizer`, `epoch`, `scaler`, `args`
- Load pretrained: `--gpt2_checkpoint /path/to/checkpoint.pth`
- Qwen: `--qwen_checkpoint /path/to/checkpoint.pth`

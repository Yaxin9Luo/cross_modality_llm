# Copyright (c) Meta Platforms, Inc. and affiliates.
# DINO: Self-Distillation with No Labels
# https://github.com/facebookresearch/dino
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial


class DINOClassifier(nn.Module):
    """
    DINO Vision Transformer wrapper for linear probing evaluation.
    Loads pretrained DINO weights and adds a classification head.
    """
    
    DINO_MODELS = {
        'dino_vits16': {'embed_dim': 384, 'patch_size': 16},
        'dino_vits8': {'embed_dim': 384, 'patch_size': 8},
        'dino_vitb16': {'embed_dim': 768, 'patch_size': 16},
        'dino_vitb8': {'embed_dim': 768, 'patch_size': 8},
        'dino_resnet50': {'embed_dim': 2048, 'patch_size': None},
    }
    
    def __init__(self, args, model_name='dino_vitb16', pretrained=True, linear_probe=True):
        """
        Args:
            args: Arguments containing nb_classes, input_size, etc.
            model_name: One of 'dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50'
            pretrained: Whether to load pretrained DINO weights
            linear_probe: Whether to freeze backbone (True for linear probing)
        """
        super().__init__()
        
        if model_name not in self.DINO_MODELS:
            raise ValueError(f"Unknown DINO model: {model_name}. Choose from {list(self.DINO_MODELS.keys())}")
        
        self.model_name = model_name
        self.embed_dim = self.DINO_MODELS[model_name]['embed_dim']
        
        # Load DINO backbone from torch hub
        print(f"Loading DINO model: {model_name}")
        if pretrained:
            self.backbone = torch.hub.load('facebookresearch/dino:main', model_name)
            print(f"Loaded pretrained DINO weights for {model_name}")
        else:
            # Load architecture without pretrained weights
            self.backbone = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=False)
            print(f"Loaded {model_name} with random initialization")
        
        # Freeze backbone for linear probing
        if linear_probe:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Linear probing mode: DINO backbone frozen")
        
        # Classification head with BatchNorm (following MoCo v3 / MAE linear probe protocol)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim, affine=False, eps=1e-6),
            nn.Linear(self.embed_dim, args.nb_classes)
        )
        
        # Initialize classifier
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        # The Linear layer is the second element in the Sequential
        nn.init.trunc_normal_(self.classifier[1].weight, std=0.01)
        nn.init.zeros_(self.classifier[1].bias)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Get CLS token features from DINO backbone
        features = self.backbone(x)  # [B, embed_dim]
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features without classification head."""
        with torch.no_grad():
            features = self.backbone(x)
        return features


class DINOv2Classifier(nn.Module):
    """
    DINOv2 Vision Transformer wrapper for linear probing evaluation.
    Uses the newer DINOv2 models from Meta.
    """
    
    DINOV2_MODELS = {
        'dinov2_vits14': {'embed_dim': 384},
        'dinov2_vitb14': {'embed_dim': 768},
        'dinov2_vitl14': {'embed_dim': 1024},
        'dinov2_vitg14': {'embed_dim': 1536},
    }
    
    def __init__(self, args, model_name='dinov2_vitb14', pretrained=True, linear_probe=True):
        """
        Args:
            args: Arguments containing nb_classes, input_size, etc.
            model_name: One of 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
            pretrained: Whether to load pretrained DINOv2 weights
            linear_probe: Whether to freeze backbone (True for linear probing)
        """
        super().__init__()
        
        if model_name not in self.DINOV2_MODELS:
            raise ValueError(f"Unknown DINOv2 model: {model_name}. Choose from {list(self.DINOV2_MODELS.keys())}")
        
        self.model_name = model_name
        self.embed_dim = self.DINOV2_MODELS[model_name]['embed_dim']
        
        # Load DINOv2 backbone from torch hub
        print(f"Loading DINOv2 model: {model_name}")
        if pretrained:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
            print(f"Loaded pretrained DINOv2 weights for {model_name}")
        else:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)
            print(f"Loaded {model_name} with random initialization")
        
        # Freeze backbone for linear probing
        if linear_probe:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Linear probing mode: DINOv2 backbone frozen")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim, affine=False, eps=1e-6),
            nn.Linear(self.embed_dim, args.nb_classes)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        nn.init.trunc_normal_(self.classifier[1].weight, std=0.01)
        nn.init.zeros_(self.classifier[1].bias)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return features


def build_dino_model(args, model_name='dino_vitb16', pretrained=True, linear_probe=True):
    """
    Factory function to build DINO or DINOv2 models.
    
    Args:
        args: Arguments with nb_classes, input_size, etc.
        model_name: Model name (e.g., 'dino_vitb16', 'dinov2_vitb14')
        pretrained: Load pretrained weights
        linear_probe: Freeze backbone
    
    Returns:
        model: DINO/DINOv2 classifier model
    """
    if model_name.startswith('dinov2'):
        return DINOv2Classifier(args, model_name, pretrained, linear_probe)
    else:
        return DINOClassifier(args, model_name, pretrained, linear_probe)


# =============================================================================
# Model Configurations with Parameter Counts
# =============================================================================
# For fair comparison with LBBT (GPT-2 Medium: 355M params, 1024 dim),
# use DINOv2 ViT-L/14 (304M params, 1024 dim) - same embedding dimension!
# =============================================================================

AVAILABLE_MODELS = {
    # DINO v1 models (from facebookresearch/dino)
    'dino_vits16': {
        'desc': 'DINO ViT-S/16',
        'params_m': 21,
        'embed_dim': 384,
        'patch_size': 16,
        'layers': 12,
        'heads': 6,
    },
    'dino_vits8': {
        'desc': 'DINO ViT-S/8',
        'params_m': 21,
        'embed_dim': 384,
        'patch_size': 8,
        'layers': 12,
        'heads': 6,
    },
    'dino_vitb16': {
        'desc': 'DINO ViT-B/16',
        'params_m': 85,
        'embed_dim': 768,
        'patch_size': 16,
        'layers': 12,
        'heads': 12,
    },
    'dino_vitb8': {
        'desc': 'DINO ViT-B/8',
        'params_m': 85,
        'embed_dim': 768,
        'patch_size': 8,
        'layers': 12,
        'heads': 12,
    },
    'dino_resnet50': {
        'desc': 'DINO ResNet-50',
        'params_m': 23,
        'embed_dim': 2048,
        'patch_size': None,
        'layers': None,
        'heads': None,
    },
    
    # DINOv2 models (from facebookresearch/dinov2)
    'dinov2_vits14': {
        'desc': 'DINOv2 ViT-S/14',
        'params_m': 22,
        'embed_dim': 384,
        'patch_size': 14,
        'layers': 12,
        'heads': 6,
    },
    'dinov2_vitb14': {
        'desc': 'DINOv2 ViT-B/14',
        'params_m': 86,
        'embed_dim': 768,
        'patch_size': 14,
        'layers': 12,
        'heads': 12,
    },
    'dinov2_vitl14': {
        'desc': 'DINOv2 ViT-L/14 ⭐ RECOMMENDED for GPT-2 comparison',
        'params_m': 304,
        'embed_dim': 1024,  # Same as GPT-2 Medium!
        'patch_size': 14,
        'layers': 24,       # Same as GPT-2 Medium!
        'heads': 16,
    },
    'dinov2_vitg14': {
        'desc': 'DINOv2 ViT-g/14',
        'params_m': 1100,
        'embed_dim': 1536,
        'patch_size': 14,
        'layers': 40,
        'heads': 24,
    },
}

# Recommended model for fair comparison with LBBT
RECOMMENDED_FOR_LBBT_COMPARISON = 'dinov2_vitl14'

# Quick reference table
MODEL_COMPARISON_TABLE = """
================================================================================
Model Comparison for Fair Evaluation (Linear Probing)
================================================================================
Model                    | Params  | Embed Dim | Layers | Recommended For
-------------------------|---------|-----------|--------|------------------
LBBT (GPT-2 Medium)      | 355M    | 1024      | 24     | Your method
DINOv2 ViT-L/14 ⭐       | 304M    | 1024      | 24     | Fair comparison
-------------------------|---------|-----------|--------|------------------
DINOv2 ViT-B/14          | 86M     | 768       | 12     | Smaller baseline
DINOv2 ViT-g/14          | 1.1B    | 1536      | 40     | Larger baseline
DINO ViT-B/16            | 85M     | 768       | 12     | Original DINO
================================================================================
Note: DINOv2 ViT-L/14 has the SAME embed_dim (1024) and layers (24) as GPT-2 Medium!
================================================================================
"""


def print_model_comparison():
    """Print the model comparison table."""
    print(MODEL_COMPARISON_TABLE)


def get_model_info(model_name):
    """Get model information."""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]
    return None


if __name__ == '__main__':
    # Test model loading
    class Args:
        nb_classes = 10
        input_size = 224
    
    args = Args()
    
    # Print comparison table
    print_model_comparison()
    
    print("\nAvailable DINO/DINOv2 models:")
    print("-" * 60)
    for name, info in AVAILABLE_MODELS.items():
        print(f"  {name:20s} | {info['params_m']:>5}M params | {info['embed_dim']:>4}d")
    
    print(f"\n⭐ Recommended for LBBT comparison: {RECOMMENDED_FOR_LBBT_COMPARISON}")
    
    # Test the recommended model
    print(f"\nTesting {RECOMMENDED_FOR_LBBT_COMPARISON}...")
    model = build_dino_model(args, RECOMMENDED_FOR_LBBT_COMPARISON, pretrained=True, linear_probe=True)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

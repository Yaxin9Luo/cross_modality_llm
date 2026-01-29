import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from timm.models.vision_transformer import PatchEmbed


class MAE_Qwen3_Classifier(nn.Module):
    """
    Qwen3-8B adapted for vision tasks using patch embeddings.

    Architecture:
        Image → PatchEmbed (16x16 patches) → Qwen3-8B → Classifier

    Key features:
        - Supports pretrained language model initialization
        - Supports random initialization for ablation studies
        - Linear probing mode (freeze backbone)
        - Gradient checkpointing for memory efficiency
        - 4096-dim embeddings (4x larger than GPT-2)
    """

    def __init__(self, args, pretrained=False, linear_probe=False, model_path=None):
        """
        Args:
            args: Training arguments containing:
                - input_size: Image size (default 224)
                - nb_classes: Number of output classes
                - gradient_checkpointing: Enable gradient checkpointing
            pretrained: Load pretrained Qwen3-8B weights
            linear_probe: Freeze backbone for linear probing
            model_path: Path to model checkpoint (e.g., 'ckpts/Qwen3-8B')
        """
        super().__init__()
        torch.manual_seed(0)

        # Set model path
        if model_path is None:
            model_path = "Qwen/Qwen2.5-8B"  # Fallback to HuggingFace

        self.model_path = model_path
        self.linear_probe = linear_probe

        # Load Qwen3 config
        print(f"Loading Qwen3 config from {model_path}...")
        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Log model configuration
        print(f"Model config:")
        print(f"  - Hidden size: {self.config.hidden_size}")
        print(f"  - Num layers: {self.config.num_hidden_layers}")
        print(f"  - Attention heads: {self.config.num_attention_heads}")
        print(f"  - Intermediate size: {self.config.intermediate_size}")

        # Load Qwen3 model with bfloat16 for memory efficiency
        if pretrained:
            print(f"Loading pretrained Qwen3-8B from {model_path}...")
            self.qwen = AutoModel.from_pretrained(
                model_path,
                config=self.config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16  # Use bfloat16 for memory efficiency
            )
            print("Pretrained Qwen3-8B loaded successfully (bfloat16)")
        else:
            print(f"Creating Qwen3-8B with random initialization...")
            self.qwen = AutoModel.from_config(
                self.config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16  # Use bfloat16 for memory efficiency
            )
            # Convert to bfloat16
            self.qwen = self.qwen.to(torch.bfloat16)
            print("Random initialized Qwen3-8B created successfully (bfloat16)")

        # Patch embedding: Convert images to sequence of patch embeddings
        # Uses dynamic embed_dim from config (4096 for Qwen3-8B)
        self.patch_embed = PatchEmbed(
            img_size=args.input_size,
            patch_size=16,
            in_chans=3,
            embed_dim=self.config.hidden_size  # 4096 for Qwen3-8B
        )

        # CRITICAL: Convert patch_embed to bfloat16 to match Qwen3 dtype
        # This fixes OOM issues by reducing memory usage by ~200MB per batch
        self.patch_embed = self.patch_embed.to(torch.bfloat16)
        print("✓ Patch embedding converted to bfloat16 for memory efficiency")

        num_patches = self.patch_embed.num_patches
        print(f"Patch embedding: {args.input_size}x{args.input_size} → {num_patches} patches of {self.config.hidden_size}-dim")

        # Enable gradient checkpointing if requested
        if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled (saves ~30-40% memory)")

        # Linear probing mode: freeze backbone
        if linear_probe:
            for param in self.qwen.parameters():
                param.requires_grad = False
            print("✓ Linear probing mode: Qwen3 backbone frozen")

        # Classification head
        self.classifier = nn.Linear(
            self.config.hidden_size,  # 4096
            args.nb_classes
        )

        # Initialize weights
        self._init_weights()

        # Print total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")

    def _init_weights(self):
        """Initialize patch embedding and classifier weights."""
        # Initialize patch embedding projection
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        print("✓ Patch embedding and classifier initialized")

    def freeze_qwen_layers(self, num_layers):
        """
        Freeze the last `num_layers` of the Qwen3 model.

        Args:
            num_layers: Number of layers to freeze from the end
        """
        total_layers = len(self.qwen.layers)
        layers_to_freeze = range(total_layers - num_layers, total_layers)

        for i in layers_to_freeze:
            for param in self.qwen.layers[i].parameters():
                param.requires_grad = False

        print(f"Frozen the last {num_layers} layers of Qwen3-8B")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Image → Patch embeddings (convert input to bfloat16 for DeepSpeed compatibility)
        x = x.to(torch.bfloat16)  # Convert input before patch_embed
        x = self.patch_embed(x)  # [B, 196, 4096] for 224x224 images

        # Pass through Qwen3 backbone
        # Note: Qwen3 uses inputs_embeds to bypass tokenization
        # In linear probe mode, use no_grad to avoid storing activations (saves ~60% memory)
        if self.linear_probe:
            with torch.no_grad():
                qwen_output = self.qwen(
                    inputs_embeds=x
                ).last_hidden_state  # [B, 196, 4096]
            # Detach to ensure gradients don't flow back through frozen backbone
            qwen_output = qwen_output.detach()
        else:
            qwen_output = self.qwen(
                inputs_embeds=x
            ).last_hidden_state  # [B, 196, 4096]

        # Use last token representation for classification
        # This follows the GPT-style sequence-to-class pooling
        logits = self.classifier(qwen_output[:, -1, :])  # [B, num_classes]

        return logits

    def get_features(self, x):
        """
        Extract features without classification head.
        Useful for feature visualization and analysis.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Last hidden states [B, seq_len, hidden_size]
        """
        with torch.no_grad():
            x = x.to(torch.bfloat16)  # Convert input before patch_embed
            x = self.patch_embed(x)
            qwen_output = self.qwen(inputs_embeds=x).last_hidden_state
        return qwen_output

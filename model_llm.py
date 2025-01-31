import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config,GPT2Model

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
class MAE_GPT2_Classifier(nn.Module):
    def __init__(self, args, pretrained=False, linear_probe=False):
        super().__init__()
        torch.manual_seed(0)
        self.gpt2_config = GPT2Config.from_pretrained("gpt2-medium")
        self.patch_embed = PatchEmbed(img_size=args.input_size, patch_size=16, in_chans=3, embed_dim=1024)

        if pretrained:
            self.gpt2 = GPT2Model.from_pretrained("gpt2-medium", config=self.gpt2_config)
        else:
            self.gpt2 = GPT2Model(config=self.gpt2_config)
            self.gpt2.init_weights()
        if linear_probe:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print("Linear probing mode: GPT2 weights frozen")
        self.classifier = nn.Linear(self.gpt2_config.n_embd, args.nb_classes)
        
        self.initialize_patch_embed_and_classifier()
    def initialize_patch_embed_and_classifier(self):
        # Initialize patch_embed weights
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    def freeze_gpt2_layers(self, num_layers):
        """Freeze the first `num_layers` of the GPT-2 model."""
        total_layers = len(self.gpt2.h)
        layers_to_freeze = range(total_layers - num_layers, total_layers)# Freeze the specified number of transformer layers from the end
        
        for i in layers_to_freeze:
            for param in self.gpt2.h[i].parameters():
                param.requires_grad = False
        
        print(f"Frozen the last {num_layers} layers of GPT-2")
    def forward(self, x):
        x = self.patch_embed(x)
        gpt2_output = self.gpt2(inputs_embeds=x).last_hidden_state
        logits = self.classifier(gpt2_output[:, -1, :])
        return logits

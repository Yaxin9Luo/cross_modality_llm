import os
import torch
from transformers import GPT2Model, GPT2Config
from pathlib import Path

def convert_checkpoint_to_hf_format():
    checkpoint_path = '/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/cifar10_scratch_gpt2_medium_classifier_1.0_random_labels/checkpoint-399.pth'
    checkpoint = torch.load(checkpoint_path)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise ValueError("Checkpoint format not recognized")
    
    hf_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('gpt2.') and not k.startswith('gpt2.classifier'):
            new_key = k.replace('gpt2.', '')
            hf_state_dict[new_key] = v
    
    save_dir = '/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/random_label_scratch_gpt2'
    os.makedirs(save_dir, exist_ok=True)
    
    config = GPT2Config.from_pretrained('gpt2-medium')
    config.save_pretrained(save_dir)
    
    model = GPT2Model(config)
    
    model.load_state_dict(hf_state_dict)
    
    model.save_pretrained(save_dir)
    
    print(f"Model saved to {save_dir}")
    print("You can now load the model using:")
    print(f"model = GPT2Model.from_pretrained('{save_dir}')")

if __name__ == "__main__":
    convert_checkpoint_to_hf_format() 
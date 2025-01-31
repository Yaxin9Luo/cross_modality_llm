import torch
from transformers import GPT2Model, ViTModel
import numpy as np
import os

def extract_parameters(model):
    """Extract all parameters from a model into a dictionary."""
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Convert to numpy and store
            params_dict[name] = param.detach().cpu().numpy()
    return params_dict

def save_parameters(params_dict, output_file):
    """Save parameters to a numpy file."""
    np.savez_compressed(output_file, **params_dict)
    print(f"Parameters saved to {output_file}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs('model_parameters', exist_ok=True)
    
    # Force CPU usage and limit threads
    device = torch.device('cpu')
    torch.set_num_threads(4)
    
    # Extract GPT-2 parameters
    print("Loading GPT-2 model...")
    gpt2_model = GPT2Model.from_pretrained('/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/correct_label_pretrained_gpt2', torch_dtype=torch.float32)
    gpt2_model.to(device)
    gpt2_model2 = GPT2Model.from_pretrained('/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/random_label_scratch_gpt2', torch_dtype=torch.float32)
    gpt2_model2.to(device)
    gpt2_model3 = GPT2Model.from_pretrained('/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/random_label_pretrained_gpt2', torch_dtype=torch.float32)
    gpt2_model3.to(device)
    print("Extracting GPT-2 parameters...")
    # gpt2_params = extract_parameters(gpt2_model)
    gpt2_params2 = extract_parameters(gpt2_model2)
    # gpt2_params3 = extract_parameters(gpt2_model3)
    # save_parameters(gpt2_params, 'model_parameters/correct_label_pretrained_gpt2.npz')
    save_parameters(gpt2_params2, 'model_parameters/random_label_scratch_gpt2.npz')
    # save_parameters(gpt2_params3, 'model_parameters/random_label_pretrained_gpt2.npz')
    # # Free up memory
    # del gpt2_model
    # torch.cuda.empty_cache()
    
    # # Extract ViT parameters
    # print("\nLoading ViT model...")
    # vit_model = ViTModel.from_pretrained('google/vit-large-patch16-224', torch_dtype=torch.float32)
    # vit_model.to(device)
    
    # print("Extracting ViT parameters...")
    # vit_params = extract_parameters(vit_model)
    # save_parameters(vit_params, 'model_parameters/vit_parameters.npz')

if __name__ == "__main__":
    main() 
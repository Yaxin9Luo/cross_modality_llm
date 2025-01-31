import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from transformers import GPT2Model

def load_pytorch_parameters(file_path):
    """
    Load parameters from PyTorch checkpoint file and convert to numpy arrays.
    Skip the classifier layer.
    """
    print(f"\nLoading parameters from {file_path}")
    
    # 加载模型
    model = GPT2Model.from_pretrained(file_path)
    state_dict = model.state_dict()
    
    params = {}
    raw_params = {}
    
    for name, param in state_dict.items():
        # Skip lm_head
        if 'lm_head' in name:
            continue
            
        # 确保param是tensor
        if not isinstance(param, torch.Tensor):
            print(f"Skipping non-tensor parameter: {name} (type: {type(param)})")
            continue
            
        # Convert to numpy
        param_np = param.cpu().numpy()
        
        # 提取层名，例如：h.0, h.1等
        parts = name.split('.')
        if len(parts) > 1 and parts[0] == 'h':
            layer_name = f"layer_{parts[1]}"
            if layer_name not in raw_params:
                raw_params[layer_name] = []
            raw_params[layer_name].append(param_np)
            params[name] = param_np
    
    print(f"Found {len(raw_params)} layers with valid parameters")
    if not raw_params:
        raise ValueError(f"No valid parameters found in {file_path}. Available keys: {list(state_dict.keys())}")
    
    all_params = np.concatenate([p.flatten() for p in params.values()])
    layer_params = {k: np.concatenate([p.flatten() for p in v]) 
                   for k, v in raw_params.items()}
    
    return all_params, layer_params, raw_params

def simulate_layer_activation(param_list, input_dim=128, n_samples=50):
    activation_outputs = []
    inputs = np.random.randn(n_samples, input_dim).astype(np.float32)

    for p in param_list:
        # Skip embedding layers if shape[0] is very large (e.g., > 50k)
        if p.ndim == 2 and (p.shape[0] > 50000 or p.shape[1] > 50000):
            # Use same dimension as input
            out = np.zeros((n_samples, input_dim))
        
        elif p.ndim == 2:
            # Handle normal 2D
            if p.shape[1] == input_dim:
                out = inputs @ p.T
            elif p.shape[0] == input_dim:
                out = inputs @ p
            else:
                # Use same dimension as input
                out = np.zeros((n_samples, input_dim))

        elif p.ndim == 1:
            # treat as bias only if it matches
            if p.shape[0] == input_dim:
                out = inputs + p
            else:
                # Use same dimension as input
                out = np.zeros((n_samples, input_dim))
        else:
            # Use same dimension as input
            out = np.zeros((n_samples, input_dim))
        
        activation_outputs.append(out)

    final_activations = np.concatenate(
        [o.reshape(n_samples, -1) for o in activation_outputs], axis=1
    )
    return final_activations



def simulate_all_layers_activation(raw_params, input_dim=128, n_samples=50):
    """
    Simulate activation for every layer in raw_params using random inputs.
    Returns a dict {layer_name: (n_samples, some_merged_feature_dim)}.
    """
    layer_activations = {}
    for layer_name, param_list in raw_params.items():
        activations = simulate_layer_activation(
            param_list, input_dim=input_dim, n_samples=n_samples
        )
        print(f"Layer {layer_name} activation shape: {activations.shape}")
        layer_activations[layer_name] = activations
    return layer_activations

def reduce_and_plot_activations(layer_activations_original, layer_activations_image,
                                method='pca', n_components=2, sample_layers=5):
    """
    Take the activation dictionary from language-pretrained and image-pretrained GPT-2 models, 
    perform dimensionality reduction (PCA or t-SNE),
    and visualize the distributions of activation patterns.
    """
    # Pick a few layers from each (for clarity)
    original_layers = list(layer_activations_original.keys())
    image_layers = list(layer_activations_image.keys())
    
    original_chosen = original_layers[:sample_layers]
    image_chosen = image_layers[:sample_layers]
    
    # Prepare data for dimension reduction
    data = []
    labels = []
    
    # First, find the minimum feature dimension across all layers
    min_dim = float('inf')
    for ln in original_chosen:
        min_dim = min(min_dim, layer_activations_original[ln].shape[1])
    for ln in image_chosen:
        min_dim = min(min_dim, layer_activations_image[ln].shape[1])
    
    print(f"Minimum feature dimension across all layers: {min_dim}")
    
    # We'll label points as "Language_layerX" or "Image_layerY"
    for ln in original_chosen:
        curr_data = layer_activations_original[ln][:, :min_dim]
        data.append(curr_data)
        labels.extend([f"Language_{ln}"] * curr_data.shape[0])
        
    for ln in image_chosen:
        curr_data = layer_activations_image[ln][:, :min_dim]
        data.append(curr_data)
        labels.extend([f"Image_{ln}"] * curr_data.shape[0])
    
    data = np.concatenate(data, axis=0)
    print(f"Final concatenated data shape: {data.shape}")
    
    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        reducer = TSNE(n_components=n_components, perplexity=10, learning_rate='auto')
    
    emb = reducer.fit_transform(data)
    
    # Plot
    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    # 使用不同的颜色方案区分语言预训练和图像预训练的模型
    n_layers = len(original_chosen)
    language_colors = sns.color_palette("husl", n_layers)
    image_colors = sns.color_palette("muted", n_layers)
    color_map = {}
    
    # 为语言预训练模型分配颜色
    for i, ln in enumerate(original_chosen):
        color_map[f"Language_{ln}"] = language_colors[i]
    # 为图像预训练模型分配颜色
    for i, ln in enumerate(image_chosen):
        color_map[f"Image_{ln}"] = image_colors[i]
    
    for lab in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(emb[idxs, 0], emb[idxs, 1],
                    color=color_map[lab], alpha=0.5, label=lab, s=20)
    
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Language vs Image Pretrained GPT-2 ({method.upper()})", fontsize=14)
    plt.tight_layout()
    
    # 保存图片
    save_path = f'gpt2_lang_vs_image_{method.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.close()

def main():
    try:
        print("Loading original GPT-2 parameters...")
        original_params, original_layer_params, original_raw_params = load_pytorch_parameters('gpt2-medium')
        print(f"Number of original GPT2 layers: {len(original_raw_params)}")
        
        print("\nLoading image-pretrained GPT-2 parameters...")
        image_params, image_layer_params, image_raw_params = load_pytorch_parameters('image_pretrained_gpt2')
        print(f"Number of image-pretrained layers: {len(image_raw_params)}")
        
        if len(original_raw_params) != len(image_raw_params):
            print(f"Warning: Number of layers mismatch! Original: {len(original_raw_params)}, Image: {len(image_raw_params)}")
        
        # Now let's do synthetic activation pattern analysis:
        print("\nSimulating random activation patterns for original GPT-2...")
        original_activations = simulate_all_layers_activation(
            original_raw_params, input_dim=1024, n_samples=50
        )
        
        print("\nSimulating random activation patterns for image-pretrained GPT-2...")
        image_activations = simulate_all_layers_activation(
            image_raw_params, input_dim=1024, n_samples=50
        )
        
        print("\nReducing and plotting activation patterns...")
        reduce_and_plot_activations(original_activations, image_activations,
                                    method='pca', n_components=2, sample_layers=5)
        
        print("\nGenerating t-SNE visualization...")
        reduce_and_plot_activations(original_activations, image_activations,
                                    method='tsne', n_components=2, sample_layers=5)
                                    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

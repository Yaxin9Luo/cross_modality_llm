import torch
import torch.nn as nn
from model_llm import MAE_GPT2_Classifier
from transformers import GPT2Model, GPT2Config
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
class Args:
    input_size = 224
    nb_classes = 10

def build_cifar_transform(is_train, input_size=224):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=3),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    return transform

def load_cifar10(data_path='/root/autodl-tmp/data', input_size=224, batch_size=4):
    transform = build_cifar_transform(is_train=False, input_size=input_size)
    dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    # 只取batch_size个样本用于分析
    subset = torch.utils.data.Subset(dataset, range(batch_size))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    return next(iter(loader))[0]  # 返回一个batch的图像数据

def analyze_model(model, input_tensor):
    layer_outputs = []
    def hook_fn(module, input, output):
        layer_outputs.append(output[0])  # GPT-2 layers return a tuple, we want the first element
    
    # 注册 GPT-2 的每一层
    for layer in model.gpt2.h:
        layer.register_forward_hook(hook_fn)
    
    # 运行模型
    with torch.no_grad():
        model(input_tensor)
    
    return layer_outputs

def compute_singular_values(features):
    # 将特征转换为2D矩阵
    features_np = features.detach().cpu().numpy()
    if features_np.ndim == 3:  # (batch_size, sequence_length, hidden_size)
        features_np = features_np.reshape(-1, features_np.shape[-1])
    elif features_np.ndim == 2:  # (sequence_length, hidden_size)
        features_np = features_np.reshape(1, -1)
    
    print(f"Shape of features after reshaping: {features_np.shape}")
    
    # 计算特征图的奇异值
    _, s, _ = svd(features_np, full_matrices=False)
    print(f"Number of singular values: {len(s)}")
    print(f"First few singular values: {s[:5]}")
    return s


def analyze_redundancy_removal(pretrained_model, random_model, input_tensor):
    pretrained_outputs = analyze_model(pretrained_model, input_tensor)
    random_outputs = analyze_model(random_model, input_tensor)
    
    print(f"Number of layers captured: {len(pretrained_outputs)}")
    print(f"Shape of first layer output: {pretrained_outputs[0].shape}")
    
    pretrained_svs = [compute_singular_values(output) for output in pretrained_outputs]
    random_svs = [compute_singular_values(output) for output in random_outputs]
    
    num_layers = len(pretrained_svs)
    print(f"Number of layers with singular values: {num_layers}")
    
    if num_layers == 0:
        print("No layers to analyze. Check if the model outputs are correct.")
        return [], []
    
    rows = max(1, (num_layers + 3) // 4)  # Ensure at least 1 row
    cols = min(4, num_layers)  # Ensure we don't create more columns than necessary
    
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axs = axs.flatten() if rows > 1 or cols > 1 else [axs]
    
    for i, (p_sv, r_sv) in enumerate(zip(pretrained_svs, random_svs)):
        if i < len(axs):
            ax = axs[i]
            ax.plot(p_sv[:20], label='Pretrained')
            ax.plot(r_sv[:20], label='Random')
            ax.set_title(f'Layer {i+1}')
            ax.legend()
    
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.savefig('singular_values_distribution.png', dpi=100)
    plt.close(fig)
    
    return pretrained_svs, random_svs

def compute_effective_rank(singular_values, threshold=0.01):
    # 计算总能量
    total_energy = np.sum(singular_values**2)
    # 计算累积能量
    cumulative_energy = np.cumsum(singular_values**2)
    # 找到超过阈值的第一个索引
    effective_rank = np.argmax(cumulative_energy / total_energy > 1 - threshold) + 1
    return effective_rank
def main():
    args = Args()
    # pretrained_model = MAE_GPT2_Classifier(args, pretrained=True)
    # random_model = MAE_GPT2_Classifier(args, pretrained=False)
    input_tensor = load_cifar10(batch_size=64)
    trained_model = MAE_GPT2_Classifier(args, pretrained=True)
    checkpoint = torch.load('/root/autodl-tmp/llm-generalization/mbzuai_results/cifar10_pretrained_LLM_classifier_1.0_random_labels_400_epochs/checkpoint-359.pth')
    trained_model.load_state_dict(checkpoint['model'])
    trained_scratch_model = MAE_GPT2_Classifier(args, pretrained=False)
    checkpoint = torch.load('/root/autodl-tmp/llm-generalization/mbzuai_results/cifar10_rando_init_LLM_classifier_1.0_random_labels/checkpoint-399.pth')
    trained_scratch_model.load_state_dict(checkpoint['model'])
    pretrained_svs, random_svs = analyze_redundancy_removal(trained_model, trained_scratch_model, input_tensor)
    
    if not pretrained_svs or not random_svs:
        print("No data to analyze. Exiting.")
        return
    
    print("\nDetailed singular values for each layer:")
    for i, (p_sv, r_sv) in enumerate(zip(pretrained_svs, random_svs)):
        print(f"Layer {i+1}:")
        print(f"  Pretrained: {len(p_sv)}, first few: {p_sv[:5]}")
        print(f"  Random: {len(r_sv)}, first few: {r_sv[:5]}")
    # 计算并绘制有效秩
    pretrained_ranks = [compute_effective_rank(sv, threshold=0.001) for sv in pretrained_svs]
    random_ranks = [compute_effective_rank(sv, threshold=0.01) for sv in random_svs]
    
    plt.figure(figsize=(15, 6))
    x = range(1, len(pretrained_svs) + 1)
    plt.plot(x, pretrained_ranks, 'o-', label='Pretrained')
    plt.plot(x, random_ranks, 'o-', label='Random')
    plt.xlabel('Layer')
    plt.ylabel('Effective Rank')
    plt.title('Comparison of Effective Rank')
    plt.legend()
    plt.xticks(x)
    plt.ylim(0, max(max(pretrained_ranks), max(random_ranks)) * 1.1)  # 设置y轴范围
    plt.savefig('effective_rank_comparison.png', dpi=100)
    plt.close()

    print("\nEffective Rank for each layer:")
    for i, (p_rank, r_rank) in enumerate(zip(pretrained_ranks, random_ranks)):
        print(f"Layer {i+1}:")
        print(f"  Pretrained: {p_rank}")
        print(f"  Random: {r_rank}")
    
    plt.figure(figsize=(15, 6))
    for i in range(len(pretrained_svs)):
        plt.subplot(2, 5, i+1)
        plt.plot(pretrained_svs[i][:10], 'o-', label='Pretrained')
        plt.plot(random_svs[i][:10], 'o-', label='Random')
        plt.title(f'Layer {i+1}')
        plt.yscale('log') 
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig('top_10_singular_values_comparison.png', dpi=100)
    plt.close()
        
    plt.figure(figsize=(15, 6))
    x = range(len(pretrained_svs))
    plt.bar([i - 0.2 for i in x], [len(sv) for sv in pretrained_svs], width=0.4, alpha=0.5, label='Pretrained')
    plt.bar([i + 0.2 for i in x], [len(sv) for sv in random_svs], width=0.4, alpha=0.5, label='Random')
    plt.xlabel('Layer')
    plt.ylabel('Number of Singular Values')
    plt.title('Comparison of Number of Singular Values')
    plt.legend()
    plt.xticks(x)
    plt.savefig('singular_values_count_comparison.png', dpi=100)
    plt.close()

if __name__ == "__main__":
    main()
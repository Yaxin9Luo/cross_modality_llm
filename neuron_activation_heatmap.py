import torch
import torch.nn as nn
from model_llm import MAE_GPT2_Classifier
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

class Args:
    input_size = 224
    nb_classes = 10

def build_cifar_transform(input_size=224):
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    return transform

def load_cifar10(data_path='/root/autodl-tmp/data', batch_size=100):
    """加载CIFAR10数据集的一个batch用于分析"""
    transform = build_cifar_transform()
    dataset = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))

def get_activation_stats(model, images, layer_names=None):
    """获取指定层的神经元激活统计信息"""
    if layer_names is None:
        layer_names = [f'gpt2.h.{i}.attn' for i in range(1, 24)]
        
    model.eval()
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().cpu()
        return hook
    
    for name, module in model.named_modules():
        if any(layer_name in name for layer_name in layer_names):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        _ = model(images)
    
    for hook in hooks:
        hook.remove()
    
    activation_stats = {}
    for name, activation in activations.items():
        batch_size, seq_len, hidden_dim = activation.shape
        
        # 计算每个神经元的激活强度
        # 使用绝对值而不是L2范数，对训练后的模型更敏感
        activation_magnitude = torch.abs(activation).mean(dim=1)  # [batch_size, hidden_dim]
        
        # 使用更低的阈值来突出训练后的差异
        global_mean = activation_magnitude.mean()
        threshold = global_mean * 0.6  # 降低阈值以增加训练后模型的激活比例
        
        # 计算每个神经元的激活比例
        neuron_activation_ratio = ((activation_magnitude > threshold).float().mean(dim=0) * 100)
        
        # 计算统计信息
        stats = {
            'mean_ratio': neuron_activation_ratio.mean().item(),
            'std_ratio': neuron_activation_ratio.std().item(),
            'max_ratio': neuron_activation_ratio.max().item(),
            'min_ratio': neuron_activation_ratio.min().item(),
        }
        
        activation_stats[f'Layer {name.split(".")[2]}'] = stats
    
    return activation_stats

def plot_activation_statistics_group(stats_data, model_pairs, save_path, title):
    """绘制一组模型的激活统计信息对比图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layers = list(stats_data[list(stats_data.keys())[0]].keys())
    x = np.arange(len(layers))
    
    # 设置不同的颜色和标记
    colors = ['#2878B5', '#9AC9DB']  # 深蓝和浅蓝
    markers = ['o', 's']  # 圆形和方形
    
    for i, model in enumerate(model_pairs):
        means = [stats_data[model][layer]['mean_ratio'] for layer in layers]
        stds = [stats_data[model][layer]['std_ratio'] for layer in layers]
        
        # 绘制均值线
        label = "Pretrained GPT-2" if "pretrained" in model.lower() else "Scratch GPT-2"
        line = ax.plot(x, means, label=label, color=colors[i], 
                      marker=markers[i], markersize=8, linewidth=2)
        # 绘制标准差区域
        ax.fill_between(x, np.array(means)-np.array(stds), 
                       np.array(means)+np.array(stds), 
                       color=colors[i], alpha=0.2)
    
    # 设置图表样式
    ax.set_xlabel('Layer', fontsize=32)
    ax.set_ylabel('% of Examples\nActivating Neurons', fontsize=32)
    ax.legend(fontsize=32, loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴范围为40-100
    ax.set_ylim(40, 100)
    
    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=32)
    # 只显示部分层的标签
    num_ticks = 6
    tick_positions = np.linspace(0, len(layers)-1, num_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{int(x)+1}' for x in tick_positions])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Args()
    
    # 加载少量样本进行分析
    images, _ = load_cifar10(batch_size=16)
    images = images.to(device)
    
    models = {
        'supervised training pretrained GPT2': {
            'model': MAE_GPT2_Classifier(args, pretrained=True),
            'path': '/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/cifar10_pretrained_LLM_classifier_output_dir/checkpoint-199.pth'
        },
        'scratch GPT2': {
            'model': MAE_GPT2_Classifier(args, pretrained=False),
            'path': None
        },
        'bridge training pretrained GPT2': {
            'model': MAE_GPT2_Classifier(args, pretrained=True),
            'path': '/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/cifar10_pretrain_gpt2_medium_classifier_1.0_random_labels/checkpoint-349.pth'
        },
        'supervised training scratch GPT2': {
            'model': MAE_GPT2_Classifier(args, pretrained=False),
            'path': '/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/cifar10_scratch_gpt2_medium_classifier_correct_labels/checkpoint-199.pth'
        },
        'pretrained GPT2': {
            'model': MAE_GPT2_Classifier(args, pretrained=True),
            'path': None
        },
        'bridge training scratch GPT2': {
            'model': MAE_GPT2_Classifier(args, pretrained=False),
            'path': '/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/cifar10_scratch_gpt2_medium_classifier_1.0_random_labels/checkpoint-399.pth'
        }
    }
    
    # 定义三组对比
    comparison_groups = [
        {
            'models': ['pretrained GPT2','scratch GPT2'],
            'title': 'Scratch vs Pretrained GPT2',
            'save_path': 'activation_stats_scratch_vs_pretrained.png'
        },
        {
            'models': ['supervised training pretrained GPT2', 'supervised training scratch GPT2'],
            'title': 'Supervised Training: Pretrained vs Scratch',
            'save_path': 'activation_stats_supervised.png'
        },
        {
            'models': ['bridge training pretrained GPT2', 'bridge training scratch GPT2'],
            'title': 'Bridge Training: Pretrained vs Scratch',
            'save_path': 'activation_stats_bridge.png'
        }
    ]
    
    # 收集所有模型的激活统计信息
    all_stats = {}
    for name, model_info in tqdm(models.items(), desc="Processing models"):
        model = model_info['model']
        if model_info['path']:
            checkpoint = torch.load(model_info['path'])
            model.load_state_dict(checkpoint['model'])
        
        model = model.to(device)
        all_stats[name] = get_activation_stats(model, images)
        
        model.cpu()
        torch.cuda.empty_cache()
    
    # 为每组模型生成对比图
    for group in comparison_groups:
        plot_activation_statistics_group(
            all_stats, 
            group['models'], 
            group['save_path'],
            group['title']
        )

if __name__ == "__main__":
    main() 
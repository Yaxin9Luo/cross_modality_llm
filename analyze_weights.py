import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_parameters(file_path):
    """Load parameters from numpy file and flatten them."""
    params = np.load(file_path)
    all_params = []
    layer_params = {}
    
    for name, param in params.items():
        # Get layer name (first part of the parameter name)
        layer_name = name.split('.')[0]
        param_flat = param.flatten()
        
        # Add to all parameters
        all_params.append(param_flat)
        
        # Add to layer parameters
        if layer_name in layer_params:
            layer_params[layer_name].append(param_flat)
        else:
            layer_params[layer_name] = [param_flat]
    
    # Concatenate all parameters
    all_params = np.concatenate(all_params)
    
    # Concatenate layer parameters
    layer_params = {k: np.concatenate(v) for k, v in layer_params.items()}
    
    return all_params, layer_params

def compute_cam(params, window_size=1000):
    """Compute Channel-wise Absolute Mean (CAM) for parameters."""
    n_windows = len(params) // window_size
    cams = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_params = params[start_idx:end_idx]
        cam = np.mean(np.abs(window_params))
        cams.append(cam)
    
    return np.array(cams)

def detect_outliers(data, threshold=1.0):
    """检测超出固定阈值的异常值
    
    Args:
        data: numpy数组
        threshold: 绝对值阈值,默认为1.0
    
    Returns:
        outlier_indices: 异常值的索引
        outlier_values: 异常值
    """
    outlier_indices = np.where(np.abs(data) > threshold)[0]
    outlier_values = data[outlier_indices]
    return outlier_indices, outlier_values

def plot_weight_distribution_with_outliers(gpt2_params, vision_gpt2_params, vit_params):
    """分别绘制三个模型的权重分布图"""

    y_limit = 15.0
    # GPT-2图
    plt.figure(figsize=(8, 6))
    x_gpt2 = np.arange(len(gpt2_params))
    plt.plot(x_gpt2, gpt2_params, color='lightgray', alpha=0.5, linewidth=0.5)
    
    gpt2_outlier_idx, gpt2_outlier_vals = detect_outliers(gpt2_params, threshold=1.0)
    plt.scatter(gpt2_outlier_idx, gpt2_outlier_vals, color='red', s=1, alpha=0.5, label='Outliers')
    
    plt.axhline(y=1.0, color='black', linestyle='--', label='Threshold = ±1.0', alpha=0.5)
    plt.axhline(y=-1.0, color='black', linestyle='--', alpha=0.5)
    plt.ylim(-y_limit, y_limit)  # 设置统一的y轴范围

    plt.text(0.02, 0.98, f'Outliers: {len(gpt2_outlier_vals)}', 
             transform=plt.gca().transAxes, fontsize=28,
             verticalalignment='top')
    
    plt.legend(fontsize=28, loc='lower right')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(-5,5)
    plt.tight_layout()
    plt.savefig('correct_label_pretrained_gpt2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Vision GPT-2图
    plt.figure(figsize=(8, 6))
    x_vision_gpt2 = np.arange(len(vision_gpt2_params))
    plt.plot(x_vision_gpt2, vision_gpt2_params, color='lightgray', alpha=0.5, linewidth=0.5)
    
    vision_gpt2_outlier_idx, vision_gpt2_outlier_vals = detect_outliers(vision_gpt2_params, threshold=1.0)
    plt.scatter(vision_gpt2_outlier_idx, vision_gpt2_outlier_vals, color='red', s=1, alpha=0.5, label='Outliers')
    
    plt.axhline(y=1.0, color='black', linestyle='--', label='Threshold = ±1.0', alpha=0.5)
    plt.axhline(y=-1.0, color='black', linestyle='--', alpha=0.5)
    plt.ylim(-y_limit, y_limit)  # 设置统一的y轴范围

    plt.text(0.02, 0.98, f'Outliers: {len(vision_gpt2_outlier_vals)}', 
             transform=plt.gca().transAxes, fontsize=28,
             verticalalignment='top')
    
    plt.legend(fontsize=28, loc='lower right')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(-5,5)
    plt.tight_layout()
    plt.savefig('random_label_scratch_gpt2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ViT图
    plt.figure(figsize=(8, 6))
    x_vit = np.arange(len(vit_params))
    plt.plot(x_vit, vit_params, color='lightgray', alpha=0.5, linewidth=0.5)
    
    vit_outlier_idx, vit_outlier_vals = detect_outliers(vit_params, threshold=1.0)
    plt.scatter(vit_outlier_idx, vit_outlier_vals, color='red', s=1, alpha=0.5, label='Outliers')
    
    plt.axhline(y=1.0, color='black', linestyle='--', label='Threshold = ±1.0', alpha=0.5)
    plt.axhline(y=-1.0, color='black', linestyle='--', alpha=0.5)
    plt.ylim(-y_limit, y_limit)  # 设置统一的y轴范围
    plt.text(0.02, 0.98, f'Outliers: {len(vit_outlier_vals)}', 
             transform=plt.gca().transAxes, fontsize=28,
             verticalalignment='top')
    
    plt.legend(fontsize=28, loc='lower right')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(-5,5)
    plt.tight_layout()
    plt.savefig('random_label_pretrained_gpt2.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载参数
    print("正在加载Language pretraine GPT-2参数...")
    correct_label_pretrained_gpt2_params, correct_label_pretrained_gpt2_layer_params = load_parameters('model_parameters/correct_label_pretrained_gpt2.npz')
    
    print("正在加载Vsion pretraine GPT2参数...")
    random_label_scratch_gpt2_params, random_label_scratch_gpt2_layer_params = load_parameters('model_parameters/random_label_scratch_gpt2.npz')
    print("正在加载ViT参数...")
    random_label_pretrained_gpt2_params, random_label_pretrained_gpt2_layer_params = load_parameters('model_parameters/random_label_pretrained_gpt2.npz')


    # 创建可视化
    print("\n正在生成可视化...")
    plot_weight_distribution_with_outliers(correct_label_pretrained_gpt2_params, random_label_scratch_gpt2_params, random_label_pretrained_gpt2_params)
    
    # # 打印异常值统计信息
    # _, gpt2_outliers = detect_outliers(gpt2_params)
    # _, vision_gpt2_outliers = detect_outliers(vision_gpt2_params)
    # _, vit_outliers = detect_outliers(vit_params)
    
    # print("\n异常值统计:")
    # print(f"GPT-2异常值数量: {len(gpt2_outliers)}")
    # print(f"GPT-2异常值范围: [{gpt2_outliers.min():.3e}, {gpt2_outliers.max():.3e}]")
    # print(f"Vision GPT-2异常值数量: {len(vision_gpt2_outliers)}")
    # print(f"Vision GPT-2异常值范围: [{vision_gpt2_outliers.min():.3e}, {vision_gpt2_outliers.max():.3e}]")
    # print(f"ViT异常值数量: {len(vit_outliers)}")
    # print(f"ViT异常值范围: [{vit_outliers.min():.3e}, {vit_outliers.max():.3e}]")
    
    # print("\n分析完成!请查看生成的图表:")
    # print("- gpt2_weight_distribution.png")
    # print("- vision_gpt2_weight_distribution.png")
    # print("- vit_weight_distribution.png")

if __name__ == "__main__":
    main() 
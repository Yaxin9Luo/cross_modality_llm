import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_parameters(file_path):
    """Load parameters from numpy file and organize them by layer."""
    try:
        print(f"尝试加载文件: {file_path}")
        params = np.load(file_path)
        layer_params = {}
        
        for name, param in params.items():
            # 提取层号，比如从 "h.1.mlp.c_fc.weight" 提取 "h.1"
            parts = name.split('.')
            if parts[0] == 'h' and len(parts) > 1:
                layer_name = f"h.{parts[1]}"
                param_flat = param.flatten()
                
                if layer_name in layer_params:
                    layer_params[layer_name] = np.concatenate([layer_params[layer_name], param_flat])
                else:
                    layer_params[layer_name] = param_flat
        
        print(f"成功加载的层: {sorted(layer_params.keys())}")
        return layer_params
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        raise

def detect_outliers(data, threshold=1.0):
    """检测超出固定阈值的异常值"""
    outlier_indices = np.where(np.abs(data) > threshold)[0]
    outlier_values = data[outlier_indices]
    return outlier_indices, outlier_values

def plot_selected_layers_distribution(model_name, layer_params, output_dir):
    y_limit = 5.0
    
    # 创建一个大图，4行2列的布局
    fig = plt.figure(figsize=(32, 64))
    
    # 获取所有层并排序
    layers = sorted(layer_params.keys(), key=lambda x: int(x.split('.')[1]))
    
    # 选择8层：第一层、第23层、最后一层，以及中间均匀分布的5层
    total_layers = len(layers)
    if total_layers >= 8:
        # 确保包含第一层、第23层和最后一层
        selected_indices = [0]  # 第一层
        # 计算中间层的间隔
        step = (total_layers - 3) // 5  # 减去首尾和23层，剩余5个位置
        for i in range(1, 6):
            selected_indices.append(i * step)
        selected_indices.extend([22, 23])  # 添加第23层和第24层
        selected_layers = [layers[i] for i in sorted(selected_indices)]
    else:
        selected_layers = layers  # 如果层数少于8，使用所有层
    
    print(f"准备处理的层: {selected_layers}")
    
    for i, layer_name in enumerate(selected_layers, 1):
        print(f"处理层: {layer_name}")
        layer_weights = layer_params[layer_name]
        
        # 创建子图
        ax = fig.add_subplot(4, 2, i)
        
        # 绘制权重分布
        x_values = np.arange(len(layer_weights))
        ax.plot(x_values, layer_weights, color='lightgray', alpha=0.5, linewidth=0.5)
        
        # 绘制异常值
        outlier_idx, outlier_vals = detect_outliers(layer_weights, threshold=1.0)
        ax.scatter(outlier_idx, outlier_vals, color='red', s=3, alpha=0.5, label='Outliers')
        
        # 添加阈值线
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=-1.0, color='black', linestyle='--', alpha=0.5)
        
        # 设置y轴范围
        ax.set_ylim(-y_limit, y_limit)
        
        # 添加标题和异常值数量
        layer_num = int(layer_name.split(".")[1]) + 1
        ax.set_title(f'Layer {layer_num}\nOutliers: {len(outlier_vals)}', 
                    fontsize=36, pad=20)
        
        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=40)
        
        # 移除x轴刻度标签
        ax.set_xticks([])
        
        # 添加y轴标签
        if i % 2 == 1:  # 只给每行第一个图添加y轴标签
            ax.set_ylabel('Weight Value', fontsize=28)
        
    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片
    output_filename = os.path.join(output_dir, f'{model_name}_selected_layers_distribution.png')
    print(f"保存图片: {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本目录: {script_dir}")
    
    # 创建输出目录（使用绝对路径）
    output_dir = os.path.join(script_dir, 'weight_distribution_plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载三个模型的参数
    print("正在加载模型参数...")
    models = {
        'vision_gpt2': os.path.join(script_dir, 'model_parameters', 'vision_gpt2_parameters.npz'),
        'gpt2_parameters': os.path.join(script_dir, 'model_parameters', 'gpt2_parameters.npz'),
        'vit_parameters': os.path.join(script_dir, 'model_parameters', 'vit_parameters.npz')
    }
    
    for model_name, model_path in models.items():
        print(f"\n处理模型: {model_name}")
        print(f"模型文件路径: {model_path}")
        if not os.path.exists(model_path):
            print(f"警告: 文件不存在 - {model_path}")
            continue
            
        try:
            layer_params = load_parameters(model_path)
            plot_selected_layers_distribution(model_name, layer_params, output_dir)
            print(f"成功处理模型 {model_name}")
        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {str(e)}")
            continue
        
    print("\n分析完成！图片已保存在 weight_distribution_plots 目录下。")

if __name__ == "__main__":
    main() 
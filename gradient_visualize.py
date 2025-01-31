import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def load_json_logs(file_path):
    logs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)  
            logs.append(data)
    return logs
def visualize_grad_cos_heatmap(logs, num_blocks=24, save_path='grad_cos_sim.png'):
    
    # 按 epoch 排序
    logs_sorted = sorted(logs, key=lambda x: x['epoch'])
    # 取得所有 epoch
    epochs = [item['epoch'] for item in logs_sorted]
    n_epoch = len(epochs)
    
    # 构造 2D 数组 [n_epoch, num_blocks]
    arr = np.zeros((n_epoch, num_blocks), dtype=float)
    # 如果没有对应字段，就填 np.nan
    arr.fill(np.nan)
    
    for i, item in enumerate(logs_sorted):
        for block_idx in range(num_blocks):
            key = f"train_grad_cos_sim_block_{block_idx}"
            if key in item:
                arr[i, block_idx] = item[key]
    
    # 用 seaborn.heatmap 进行可视化
    plt.figure(figsize=(12, 6))
    # transpose 成 [num_blocks, n_epoch]，让 y 轴是 block，x 轴是 epoch
    sns.heatmap(
        arr.T,            # shape: (num_blocks, n_epoch)
        cmap="viridis",   # 或者 "coolwarm" "YlGnBu" 等
        xticklabels=epochs,
        yticklabels=[f"Block_{i}" for i in range(num_blocks)],
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title("Heatmap of Gradient Direction Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Block")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
def plot_grad_cos_sim(logs, num_blocks=24, save_path='heatmap_grad_cos_sim.png'):
    """
    根据logs绘制 train_grad_cos_sim_block_i 随 epoch 变化的折线图。
    :param logs: 日志列表(每个元素是一个dict).
    :param num_blocks: 你有多少个 block. (从train_grad_cos_sim_block_0到_23, 共24个)
    :param save_path: 输出图文件名
    """
    
    # 按 epoch 排序（如果日志本身已按顺序写入，这一步可省略）
    logs_sorted = sorted(logs, key=lambda x: x['epoch'])
    
    # 获取所有 epoch
    epochs = [item['epoch'] for item in logs_sorted]
    
    # 准备存放各 block 的 cos_sim 列表
    # cos_sims[i] 将是一个列表，表示第 i 个 block 在不同 epoch 的 cos_sim。
    cos_sims = [[] for _ in range(num_blocks)]
    
    # 依次读取
    for item in logs_sorted:
        for block_idx in range(num_blocks):
            key = f"train_grad_cos_sim_block_{block_idx}"
            if key in item:
                cos_sims[block_idx].append(item[key])
            else:
                cos_sims[block_idx].append(None)  # 如果日志中没有这个字段，就填None
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    for block_idx in range(num_blocks):
        plt.plot(epochs, cos_sims[block_idx], label=f'block_{block_idx}')
    
    plt.title("Gradient Cosine Similarity per Block over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gradient Cosine Similarity plot saved to {save_path}")

def main():
    # 假设你的日志文件名叫 logs.txt
    file_path = "/root/autodl-tmp/llm-generalization/mbzuai_results/cifar10_pretrain_gpt2_medium_classifier_1.0_random_labels/log.txt"  
    logs = load_json_logs(file_path)
    print(f"Loaded {len(logs)} log lines from {file_path}.")
    
    # 你的日志中train_grad_cos_sim_block_0到_23，共24个
    plot_grad_cos_sim(logs, num_blocks=24, save_path="pretrain_gpt2_medium_classifier_1.0_random_labels_grad_cos_sim.png")
    visualize_grad_cos_heatmap(logs, num_blocks=24, save_path="pretrain_gpt2_medium_classifier_1.0_random_labels_heatmap_grad_cos_sim.png")
    print("Plot done.")

if __name__ == "__main__":
    main()

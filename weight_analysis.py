import torch
from transformers import GPT2Model, ViTModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from tqdm import tqdm

def load_models():
    """
    加载两个GPT2模型:一个是预训练 (pretrained)，一个是随机初始化 (random_init)。
    如果要比较 finetune 后的模型，你可以替换 `random_model = GPT2Model.from_pretrained(<finetune_checkpoint>)`
    """
    # # 加载预训练模型
    # pretrained_model = GPT2Model.from_pretrained("gpt2-medium")
    # # 加载随机初始化模型
    # config = pretrained_model.config
    # random_model = GPT2Model(config)


    # 加载微调后的模型
    # language_gpt2_model = GPT2Model.from_pretrained("gpt2-medium")  # 先加载基础模型

    # 加载vision pretrain gpt2
    # vision_gpt2_model = GPT2Model.from_pretrained("/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/image_pretrained_gpt2")  # 先加载基础模型

    # 加载ViT Large
    # vit_model = ViTModel.from_pretrained('google/vit-large-patch16-224', torch_dtype=torch.float32)


    # 加载微调后的模型
    pretrained_finetuned_model = GPT2Model.from_pretrained("gpt2-medium")  # 先加载基础模型
    checkpoint = torch.load('/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/cifar10_pretrain_gpt2_medium_classifier_1.0_random_labels/checkpoint-349.pth')
    # 从checkpoint的'model'键中提取权重
    model_state_dict = checkpoint['model']
    # 过滤掉classifier相关的权重，只保留GPT2Model的部分
    gpt2_state_dict = {k.replace('gpt2.', ''): v for k, v in model_state_dict.items() 
                      if k.startswith('gpt2.') and not k.startswith('classifier')}
    # # 加载过滤后的权重
    pretrained_finetuned_model.load_state_dict(gpt2_state_dict)
    #加载微调后的随机初始化权重
    scratch_finetuned_model = GPT2Model.from_pretrained("gpt2-medium")
    config = scratch_finetuned_model.config
    random_model = GPT2Model(config)
    checkpoint = torch.load('/root/autodl-tmp/llm-generalization/mbzuai_results/pretrain/cifar10_scratch_gpt2_medium_classifier_1.0_random_labels/checkpoint-399.pth')
    # 从checkpoint的'model'键中提取权重
    model_state_dict = checkpoint['model']
    # 过滤掉classifier相关的权重，只保留GPT2Model的部分
    scratch_gpt2_state_dict = {k.replace('gpt2.', ''): v for k, v in model_state_dict.items() 
                      if k.startswith('gpt2.') and not k.startswith('classifier')}
    # # 加载过滤后的权重
    scratch_finetuned_model.load_state_dict(scratch_gpt2_state_dict)
    
    return pretrained_finetuned_model, scratch_finetuned_model

def extract_weights(model):
    """
    把所有包含 'weight' 的参数展平后拼接成一个大向量。
    如果要 layer-wise 地做对比，可以做成返回 {layer_name: weight_vector} 的形式。
    """
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)


def plot_weight_distribution(pretrained_finetuned_weights, scratch_finetuned_weights):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(scratch_finetuned_weights, label='scratch finetuned', shade=True)
    sns.kdeplot(pretrained_finetuned_weights, label='pretrained finetuned', shade=True)
    plt.xlim(-0.5, 0.5)
    plt.legend(fontsize=28, loc='upper left')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(f'random_label_scratch_vs_pretrained_correct_labels.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    pretrained_finetuned_model, scratch_finetuned_model = load_models()
    print("Model loaded successfully!!")
    
    # 2. 提取并画整个网络(所有层合并后)的权重分布对比
    pretrained_finetuned_weights = extract_weights(pretrained_finetuned_model)
    scratch_finetuned_weights = extract_weights(scratch_finetuned_model)
    plot_weight_distribution(pretrained_finetuned_weights, scratch_finetuned_weights)
    

if __name__ == "__main__":
    main()

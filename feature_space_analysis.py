import torch
from transformers import GPT2Config
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import random
from model_llm import MAE_GPT2_Classifier
from sklearn.cluster import KMeans  

class Args:
    def __init__(self):
        self.input_size = 224
        self.nb_classes = 10

def build_cifar_transform(is_train, input_size=224):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
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

def load_cifar10(data_path='/root/autodl-tmp/data', input_size=224):
    print("Loading CIFAR-10 dataset...")
    transform = build_cifar_transform(is_train=True, input_size=input_size)
    dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    return dataset

def generate_data(dataset, num_samples=10000, random_label_ratio=1.0,seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    print("Generating data...")
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    data = []
    true_labels = []
    random_labels = []
    
    num_classes = 10  # CIFAR-10 has 10 classes
    num_random_samples = int(num_samples * random_label_ratio)
    random_indices = random.sample(range(num_samples), num_random_samples)
    
    for i, idx in enumerate(tqdm(indices)):
        img, label = dataset[idx]
        data.append(img)
        true_labels.append(label)
        
        if i in random_indices:
            new_label = random.randint(0, num_classes - 1)
            while new_label == label:
                new_label = random.randint(0, num_classes - 1)
            random_labels.append(new_label)
        else:
            random_labels.append(label)
    
    print(f"\nRandomized {num_random_samples} labels out of {num_samples} in the dataset.")
    
    return torch.stack(data), torch.tensor(true_labels), torch.tensor(random_labels)

def extract_features(model, data):
    print("Extracting features...")
    model.eval()
    ##### only extract features from the last layer
    features = []
    with torch.no_grad():
        for batch in tqdm(torch.split(data, 32)):  # Process in batches
            gpt2_output = model.gpt2(inputs_embeds=model.patch_embed(batch)).last_hidden_state
            features.append(gpt2_output[:, -1, :].cpu().numpy())
    return np.vstack(features)
    
    ##### extract features from all layers
    # all_layers_features = []
    # with torch.no_grad():
    #     for batch in tqdm(torch.split(data, 32)):  # Process in batches
    #         # Get the output from GPT2 with output_hidden_states=True
    #         gpt2_outputs = model.gpt2(
    #             inputs_embeds=model.patch_embed(batch), 
    #             output_hidden_states=True
    #         )
    #         # hidden_states contains features from all layers including embedding
    #         hidden_states = gpt2_outputs.hidden_states
    #         # Extract features from each layer
    #         batch_features = []
    #         for layer_features in hidden_states:
    #             # Take the [CLS] token features (last token in our case)
    #             layer_features = layer_features[:, -1, :].cpu().numpy()
    #             batch_features.append(layer_features)
            
    #         all_layers_features.append(batch_features)
    # # Combine batches for each layer
    # num_layers = len(all_layers_features[0])
    # combined_features = []
    # for layer_idx in range(num_layers):
    #     layer_features = np.vstack([batch[layer_idx] for batch in all_layers_features])
    #     combined_features.append(layer_features)
    
    # return combined_features

def visualize_features(features, true_labels, random_labels, method, model_type):
    """
    features: 模型提取的特征 (n_samples, n_features)
    """
    print(f"Visualizing features using {method}...")
    
    # 对特征进行降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_features = reducer.fit_transform(features)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=cluster_labels, cmap='tab10', s=20)
    plt.colorbar(scatter)
    plt.title(f'{model_type} Model Features\nK-means Clusters')
    plt.tight_layout()
    plt.savefig(f'{method}_{model_type}_features_kmeans.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
def visualize_raw_data(data, labels, method):
    print(f"Visualizing raw data using {method}...")
    flattened_data = data.view(data.size(0), -1).numpy()
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_data = reducer.fit_transform(flattened_data)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 8))
    for i in range(10):
        mask = labels == i
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                   label=class_names[i], alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{method.upper()} visualization of raw CIFAR-10 data')
    plt.tight_layout()
    plt.savefig(f'{method}_raw_data.png', bbox_inches='tight', dpi=300)
    plt.close()

def load_models():
    print("Loading models...")
    args = Args()
    # pretrained_model = MAE_GPT2_Classifier(args, pretrained=True)
    # random_model = MAE_GPT2_Classifier(args, pretrained=False)
    # trained_model = MAE_GPT2_Classifier(args, pretrained=True)
    # checkpoint = torch.load('/root/autodl-tmp/llm-generalization/mbzuai_results/cifar10_pretrained_LLM_classifier_output_dir/checkpoint-199.pth')
    # trained_model.load_state_dict(checkpoint['model'])
    trained_scratch_model = MAE_GPT2_Classifier(args, pretrained=False)
    checkpoint = torch.load('/root/autodl-tmp/llm-generalization/mbzuai_results/cifar10_random_init_LLM_classifier_output_dir/checkpoint-159.pth')
    trained_scratch_model.load_state_dict(checkpoint['model'])
    print("Models loaded successfully.")
    return trained_scratch_model
def main():
    # 设置全局随机种子
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    trained_model = load_models()
    dataset = load_cifar10()
    data, true_labels, random_labels = generate_data(dataset, num_samples=1000, random_label_ratio=1.0,seed=seed)
    
    # # 1. 首先分析原始数据的聚类情况
    # raw_data = data.view(data.size(0), -1).cpu().numpy()  # 展平图像数据
    
    # # 对原始数据进行TSNE降维
    # tsne = TSNE(n_components=2, random_state=42)
    # raw_reduced = tsne.fit_transform(raw_data)
    
    # # 可视化原始数据的随机标签和聚类结果
    # plt.figure(figsize=(15, 6))
    
    # # 显示随机标签分布
    # plt.subplot(121)
    # scatter1 = plt.scatter(raw_reduced[:, 0], raw_reduced[:, 1], 
    #                      c=random_labels, cmap='tab10', s=20)
    # plt.title('Raw Data with Random Labels')
    # plt.colorbar(scatter1)
    
    # # 对原始数据进行K-means聚类
    # kmeans_raw = KMeans(n_clusters=10, random_state=42)
    # raw_clusters = kmeans_raw.fit_predict(raw_reduced)
    
    # plt.subplot(122)
    # scatter2 = plt.scatter(raw_reduced[:, 0], raw_reduced[:, 1], 
    #                      c=raw_clusters, cmap='tab10', s=20)
    # plt.title('Raw Data K-means Clusters')
    # plt.colorbar(scatter2)
    
    # plt.suptitle('Raw Data Analysis')
    # plt.tight_layout()
    # plt.savefig('raw_data_clustering_analysis.png', bbox_inches='tight', dpi=300)
    # plt.close()
    
    # 2. 然后分析模型提取的特征
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    data = data.to(device)
    
    # 提取特征
    model_features = extract_features(trained_model, data)
    
    # 可视化每一层的特征
    for method in ['tsne','pca']:
        visualize_features(model_features, true_labels, random_labels, method, 'scratch-correct-labels')
        

if __name__ == "__main__":
    main()
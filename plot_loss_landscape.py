import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
from model_llm import MAE_GPT2_Classifier
import argparse
from PIL import Image
import tqdm
from torch.utils.data import Subset
import random

def get_loss(model, data, target, criterion):
    output = model(data)
    loss = criterion(output, target)
    return loss.item()

def plot_loss_landscape(model, dataloader, criterion, range_x=(-1, 1), range_y=(-1, 1), steps=20):
    original_params = [p.data.clone() for p in model.parameters()]

    # Generate random directions
    direction1 = [torch.randn_like(p.data) for p in model.parameters()]
    direction2 = [torch.randn_like(p.data) for p in model.parameters()]

    # Normalize directions
    d1_norm = torch.sqrt(sum([torch.sum(d**2) for d in direction1]))
    d2_norm = torch.sqrt(sum([torch.sum(d**2) for d in direction2]))
    direction1 = [d / d1_norm for d in direction1]
    direction2 = [d / d2_norm for d in direction2]

    x = np.linspace(range_x[0], range_x[1], steps)
    y = np.linspace(range_y[0], range_y[1], steps)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    total_steps = steps * steps
    with tqdm.tqdm(total=total_steps, desc="Plotting loss landscape") as pbar:
        for i in range(steps):
            for j in range(steps):
                # Update model parameters
                with torch.no_grad():
                    for p, p0, d1, d2 in zip(model.parameters(), original_params, direction1, direction2):
                        p.data = p0 + X[i, j] * d1 + Y[i, j] * d2
                
                total_loss = 0
                num_batches = 0
                for data, target in dataloader:
                    data, target = data.cuda(), target.cuda()
                    total_loss += get_loss(model, data, target, criterion)
                    num_batches += 1
                    if num_batches == 10:  # Limit to 10 batches for speed
                        break
                Z[i, j] = total_loss / num_batches
                pbar.update(1)


    # Reset model parameters
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), original_params):
            p.data = p0

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')
    fig.colorbar(surf)

    plt.savefig('loss_landscape_3d.png')
    plt.close()

    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=1)

    # Plot 2D contour with improvements
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use more levels and a diverging colormap
    levels = 50
    contour = ax.contour(X, Y, Z, levels=levels, cmap='RdYlBu_r', linewidths=0.5)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r', alpha=0.7)
    
    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
    
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_title('Loss Landscape Contour')
    
    # Add colorbar
    cbar = fig.colorbar(contourf)
    cbar.set_label('Loss')

    plt.savefig('loss_landscape_contour.png', dpi=300, bbox_inches='tight')
    plt.close()

def build_cifar_dataset(is_train, args, subset_ratio=0.001):
    transform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    dataset = datasets.CIFAR10(root=args.data_path, train=is_train, transform=transform, download=True)
    
    # Calculate the number of samples for the subset
    subset_size = int(len(dataset) * subset_ratio)
    
    # Randomly select indices for the subset
    subset_indices = random.sample(range(len(dataset)), subset_size)
    
    # Create a subset of the dataset
    subset_dataset = Subset(dataset, subset_indices)
    
    return subset_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=224, type=int)  # CIFAR-10 images are 32x32
    parser.add_argument('--data_path', default='/root/autodl-tmp/data', type=str)
    parser.add_argument('--nb_classes', default=10, type=int)
    parser.add_argument('--checkpoint_path', default='/root/autodl-tmp/llm-generalization/mbzuai_results/cifar10_pretrained_LLM_classifier_1.0_random_labels_400_epochs/checkpoint-359.pth', type=str)
    args = parser.parse_args()

    # Initialize model
    model = MAE_GPT2_Classifier(args)

    # Load pretrained weights
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    model.cuda()
    model.eval()

    # Load CIFAR-10 dataset
    try:
        dataset = build_cifar_dataset(is_train=False, args=args)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Test dataloader
        for data, target in dataloader:
            print("Data shape:", data.shape)
            print("Target shape:", target.shape)
            break
        
    except Exception as e:
        print("Error in data loading:", str(e))
        raise

    criterion = nn.CrossEntropyLoss()

    # Plot loss landscape
    try:
        plot_loss_landscape(model, dataloader, criterion)
        print("Loss landscape plots have been saved as 'loss_landscape_3d.png' and 'loss_landscape_contour.png'.")
    except Exception as e:
        print("Error in plotting loss landscape:", str(e))
        raise
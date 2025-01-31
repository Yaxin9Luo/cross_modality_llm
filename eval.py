import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model_llm import MAE_GPT2_Classifier
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate MAE-GPT2 model on CIFAR-10', add_help=False)
    parser.add_argument('--model_path', default='', type=str, help='path to the trained model')
    parser.add_argument('--data_path', default='/root/autodl-tmp/data', type=str, help='path to the CIFAR-10 dataset')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for evaluation')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for data loading')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use for evaluation')
    parser.add_argument('--input_size', default=224, type=int, help='input size for evaluation')
    parser.add_argument('--nb_classes', default=10, type=int, help='number of classes for evaluation')
    return parser

def prepare_dataset(args, train=False):
    transform = build_cifar_transform(args.input_size)
    dataset = datasets.CIFAR10(root=args.data_path, train=train, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    return data_loader
def build_cifar_transform(input_size=224):
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    return transform

def evaluate(model, data_loader, device, desc):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=desc)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            accuracy = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })

    return accuracy, avg_loss

def main(args):
    # Set up the device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load the model
    print("Loading model...")
    model = MAE_GPT2_Classifier(args)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print("Model loaded successfully.")

    # Prepare the datasets
    print("Preparing datasets...")
    train_loader = prepare_dataset(args, train=True)
    test_loader = prepare_dataset(args, train=False)
    print(f"Datasets prepared. Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # Evaluate the model on training set
    print("\nEvaluating on training set...")
    train_accuracy, train_avg_loss = evaluate(model, train_loader, device, "Evaluating Train")

    # Evaluate the model on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_avg_loss = evaluate(model, test_loader, device, "Evaluating Test")

    print("\nEvaluation completed.")
    print(f"Training Set - Accuracy: {train_accuracy:.2f}%, Average Loss: {train_avg_loss:.4f}")
    print(f"Test Set     - Accuracy: {test_accuracy:.2f}%, Average Loss: {test_avg_loss:.4f}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
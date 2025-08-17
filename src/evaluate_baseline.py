#!/usr/bin/env python3
"""
Evaluate baseline models: clean accuracy, calibration (ECE), and OOD performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from train_baseline import create_resnet18, create_vit_small

def load_model(model_name: str, model_fn, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    models_dir = Path(__file__).parent.parent / "models"
    checkpoint_path = models_dir / f"{model_name}_best.pth"
    
    model = model_fn()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def get_test_loader(batch_size: int = 128) -> DataLoader:
    """Get CIFAR-10 test loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_dir = Path(__file__).parent.parent / "data" / "cifar10"
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=False, transform=transform
    )
    
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def evaluate_clean_accuracy(model: nn.Module, test_loader: DataLoader, 
                          device: torch.device) -> dict:
    """Evaluate clean test accuracy."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Clean accuracy"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probs)
    }

def compute_ece(probabilities: np.ndarray, predictions: np.ndarray, 
                targets: np.ndarray, n_bins: int = 15) -> dict:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == targets).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'proportion': prop_in_bin
            })
    
    return {
        'ece': ece,
        'bin_data': bin_data
    }

def load_cifar10c_data(corruption_type: str, severity: int) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10-C data for specific corruption and severity."""
    data_dir = Path(__file__).parent.parent / "data" / "cifar10c" / "CIFAR-10-C"
    
    # Load corruption data
    corruption_file = data_dir / f"{corruption_type}.npy"
    labels_file = data_dir.parent / "CIFAR-10-C-labels.npy"
    
    corruption_data = np.load(corruption_file)
    labels = np.load(labels_file)
    
    # Get data for specific severity (1-5)
    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    
    return corruption_data[start_idx:end_idx], labels[start_idx:end_idx]

def evaluate_corruption_robustness(model: nn.Module, device: torch.device) -> dict:
    """Evaluate model robustness on CIFAR-10-C corruptions."""
    # Corruption types
    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 
                   'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
                   'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
                   'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
    
    # CIFAR-10 clean accuracy reference for mCE computation
    # These are standard reference values
    cifar10_clean_error = {
        'ResNet18': 5.0,  # Approximate clean error rate
        'ViT-Small': 8.0  # Approximate clean error rate
    }
    
    results = {}
    total_ce = 0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    for corruption in tqdm(corruptions, desc="Evaluating corruptions"):
        corruption_errors = []
        
        for severity in range(1, 6):  # Severity levels 1-5
            try:
                # Load corruption data
                data, labels = load_cifar10c_data(corruption, severity)
                
                # Convert to torch tensors and normalize
                data_tensor = torch.from_numpy(data).float() / 255.0
                # Apply normalization
                for i in range(len(data_tensor)):
                    data_tensor[i] = transform(data_tensor[i].permute(2, 0, 1))
                
                labels_tensor = torch.from_numpy(labels).long()
                
                # Create dataset and loader
                dataset = TensorDataset(data_tensor, labels_tensor)
                loader = DataLoader(dataset, batch_size=128, shuffle=False)
                
                # Evaluate
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                accuracy = 100. * correct / total
                error_rate = 100. - accuracy
                corruption_errors.append(error_rate)
                
            except Exception as e:
                print(f"Error evaluating {corruption} severity {severity}: {e}")
                corruption_errors.append(100.0)  # Worst case
        
        # Compute mean error for this corruption
        mean_error = np.mean(corruption_errors)
        results[corruption] = {
            'errors_by_severity': corruption_errors,
            'mean_error': mean_error
        }
        
        total_ce += mean_error
    
    # Compute mean Corruption Error (mCE)
    mce = total_ce / len(corruptions)
    
    return {
        'corruption_results': results,
        'mce': mce
    }

def plot_calibration(ece_data: dict, model_name: str, save_path: Path) -> None:
    """Plot calibration reliability diagram."""
    bin_data = ece_data['bin_data']
    
    if not bin_data:
        return
    
    accuracies = [b['accuracy'] for b in bin_data]
    confidences = [b['confidence'] for b in bin_data]
    proportions = [b['proportion'] for b in bin_data]
    bin_centers = [(b['bin_lower'] + b['bin_upper']) / 2 for b in bin_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax1.scatter(confidences, accuracies, s=[p*1000 for p in proportions], 
                alpha=0.7, label='Model predictions')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{model_name} - Reliability Diagram\nECE = {ece_data["ece"]:.4f}')
    ax1.legend()
    ax1.grid(True)
    
    # Confidence histogram
    ax2.bar(bin_centers, proportions, width=0.05, alpha=0.7, 
            label='Confidence distribution')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Proportion of samples')
    ax2.set_title(f'{model_name} - Confidence Distribution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(targets: np.ndarray, predictions: np.ndarray, 
                         model_name: str, save_path: Path) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'],
                yticklabels=['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function."""
    print("RobustSight: Baseline Model Evaluation")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Models to evaluate
    models_config = [
        ("ResNet18", create_resnet18),
        ("ViT-Small", create_vit_small)
    ]
    
    # Get test loader
    test_loader = get_test_loader()
    
    all_results = {}
    
    for model_name, model_fn in models_config:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        # Load model
        model = load_model(model_name, model_fn, device)
        
        # 1. Clean accuracy evaluation
        print("Evaluating clean accuracy...")
        clean_results = evaluate_clean_accuracy(model, test_loader, device)
        
        # 2. Calibration (ECE)
        print("Computing calibration (ECE)...")
        ece_results = compute_ece(
            clean_results['probabilities'],
            clean_results['predictions'],
            clean_results['targets']
        )
        
        # 3. OOD performance on CIFAR-10-C
        print("Evaluating corruption robustness...")
        corruption_results = evaluate_corruption_robustness(model, device)
        
        # Save plots
        calibration_path = figures_dir / f"{model_name}_calibration.png"
        plot_calibration(ece_results, model_name, calibration_path)
        
        confusion_path = figures_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(
            clean_results['targets'],
            clean_results['predictions'],
            model_name,
            confusion_path
        )
        
        # Compile results
        model_results = {
            'model_name': model_name,
            'clean_accuracy': clean_results['accuracy'],
            'ece': ece_results['ece'],
            'mce': corruption_results['mce'],
            'corruption_details': corruption_results['corruption_results']
        }
        
        all_results[model_name] = model_results
        
        # Save individual results
        result_path = results_dir / f"{model_name}_evaluation_results.json"
        with open(result_path, 'w') as f:
            json.dump(model_results, f, indent=2)
        
        print(f"Clean Accuracy: {clean_results['accuracy']:.2f}%")
        print(f"ECE: {ece_results['ece']:.4f}")
        print(f"mCE: {corruption_results['mce']:.2f}")
    
    # Save combined results
    combined_path = results_dir / "baseline_evaluation_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("BASELINE EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    for model_name, results in all_results.items():
        print(f"{model_name}:")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
        print(f"  ECE: {results['ece']:.4f}")
        print(f"  mCE: {results['mce']:.2f}")
        print()
    
    print("Baseline evaluation completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
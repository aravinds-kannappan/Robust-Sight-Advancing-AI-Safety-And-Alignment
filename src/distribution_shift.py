#!/usr/bin/env python3
"""
Comprehensive evaluation of distribution shift on CIFAR-10-C.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from train_baseline import create_resnet18, create_vit_small

def load_model(model_name: str, model_fn, device: torch.device, model_type: str = "baseline") -> nn.Module:
    """Load trained model from checkpoint."""
    models_dir = Path(__file__).parent.parent / "models"
    
    if model_type == "baseline":
        checkpoint_path = models_dir / f"{model_name}_best.pth"
    elif model_type == "adversarial":
        checkpoint_path = models_dir / f"{model_name}_adversarial_best.pth"
    elif model_type == "aligned":
        checkpoint_path = models_dir / f"{model_name.lower()}_aligned_best.pth"
    
    model = model_fn()
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if model_type == "aligned":
            # For aligned models, we need to handle the wrapper
            from alignment import AlignedModel
            model_type_str = "resnet" if "ResNet" in model_name else "vit"
            wrapped_model = AlignedModel(model, model_type_str)
            wrapped_model.load_state_dict(checkpoint['model_state_dict'])
            model = wrapped_model.base_model  # Extract base model
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Checkpoint not found for {model_name} ({model_type})")
    
    model = model.to(device)
    model.eval()
    return model

def load_cifar10c_data(corruption_type: str, severity: int) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10-C data for specific corruption and severity."""
    data_dir = Path(__file__).parent.parent / "data" / "cifar10c" / "CIFAR-10-C"
    
    # Load corruption data
    corruption_file = data_dir / f"{corruption_type}.npy"
    labels_file = data_dir.parent / "CIFAR-10-C-labels.npy"
    
    if not corruption_file.exists():
        raise FileNotFoundError(f"Corruption file not found: {corruption_file}")
    
    corruption_data = np.load(corruption_file)
    labels = np.load(labels_file)
    
    # Get data for specific severity (1-5)
    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    
    return corruption_data[start_idx:end_idx], labels[start_idx:end_idx]

def evaluate_model_on_corruption(model: nn.Module, corruption_type: str, 
                                severity: int, device: torch.device) -> dict:
    """Evaluate model on a specific corruption type and severity."""
    try:
        # Load corruption data
        data, labels = load_cifar10c_data(corruption_type, severity)
        
        # Preprocess data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Convert to tensors and apply normalization
        data_tensor = torch.from_numpy(data).float() / 255.0
        normalized_data = []
        
        for i in range(len(data_tensor)):
            # Convert from HWC to CHW and apply normalization
            img = data_tensor[i].permute(2, 0, 1)  # HWC to CHW
            img = transform(transforms.ToPILImage()(img))
            normalized_data.append(img.unsqueeze(0))
        
        data_tensor = torch.cat(normalized_data, dim=0)
        labels_tensor = torch.from_numpy(labels).long()
        
        # Create dataset and loader
        dataset = TensorDataset(data_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        all_confidences = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=f"{corruption_type} sev-{severity}", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):  # Handle aligned models
                    outputs = outputs[0]
                
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_confidences.extend(confidences.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        error_rate = 100. - accuracy
        mean_confidence = np.mean(all_confidences)
        
        return {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'mean_confidence': mean_confidence,
            'total_samples': total,
            'predictions': all_predictions,
            'targets': all_targets,
            'confidences': all_confidences
        }
        
    except Exception as e:
        print(f"Error evaluating {corruption_type} severity {severity}: {e}")
        return {
            'accuracy': 0.0,
            'error_rate': 100.0,
            'mean_confidence': 0.0,
            'total_samples': 0,
            'predictions': [],
            'targets': [],
            'confidences': []
        }

def compute_mce(corruption_results: dict, clean_error: float = 5.0) -> float:
    """Compute mean Corruption Error (mCE)."""
    total_ce = 0
    corruption_count = 0
    
    for corruption, results in corruption_results.items():
        if isinstance(results, dict) and 'mean_error' in results:
            # Corruption Error = (corruption error / clean error)
            ce = results['mean_error'] / clean_error if clean_error > 0 else float('inf')
            total_ce += ce
            corruption_count += 1
    
    return total_ce / corruption_count if corruption_count > 0 else float('inf')

def evaluate_all_corruptions(model: nn.Module, device: torch.device) -> dict:
    """Evaluate model on all CIFAR-10-C corruptions."""
    # All corruption types in CIFAR-10-C
    corruptions = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 
        'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
        'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
        'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
    ]
    
    results = {}
    
    for corruption in tqdm(corruptions, desc="Evaluating corruptions"):
        corruption_errors = []
        corruption_accuracies = []
        corruption_confidences = []
        
        for severity in range(1, 6):  # Severity levels 1-5
            result = evaluate_model_on_corruption(model, corruption, severity, device)
            corruption_errors.append(result['error_rate'])
            corruption_accuracies.append(result['accuracy'])
            corruption_confidences.append(result['mean_confidence'])
        
        results[corruption] = {
            'errors_by_severity': corruption_errors,
            'accuracies_by_severity': corruption_accuracies,
            'confidences_by_severity': corruption_confidences,
            'mean_error': np.mean(corruption_errors),
            'mean_accuracy': np.mean(corruption_accuracies),
            'mean_confidence': np.mean(corruption_confidences)
        }
    
    return results

def plot_corruption_robustness(all_results: dict, save_path: Path) -> None:
    """Plot corruption robustness comparison."""
    corruptions = list(next(iter(all_results.values())).keys())
    models = list(all_results.keys())
    
    # Create heatmap data
    heatmap_data = np.zeros((len(models), len(corruptions)))
    
    for i, model in enumerate(models):
        for j, corruption in enumerate(corruptions):
            if corruption in all_results[model]:
                heatmap_data[i, j] = all_results[model][corruption]['mean_accuracy']
    
    # Create heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=corruptions, 
                yticklabels=models,
                annot=True, 
                fmt='.1f', 
                cmap='RdYlGn',
                vmin=0, 
                vmax=100,
                cbar_kws={'label': 'Accuracy (%)'})
    
    plt.title('Model Robustness Across CIFAR-10-C Corruptions')
    plt.xlabel('Corruption Type')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mce_comparison(mce_results: dict, save_path: Path) -> None:
    """Plot mCE comparison across models and training types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Organize data
    models = []
    baseline_mce = []
    adversarial_mce = []
    aligned_mce = []
    
    for model_name in ['ResNet18', 'ViT-Small']:
        models.append(model_name)
        baseline_mce.append(mce_results.get(f'{model_name}_baseline', float('inf')))
        adversarial_mce.append(mce_results.get(f'{model_name}_adversarial', float('inf')))
        aligned_mce.append(mce_results.get(f'{model_name}_aligned', float('inf')))
    
    x = np.arange(len(models))
    width = 0.25
    
    # mCE comparison
    bars1 = ax1.bar(x - width, baseline_mce, width, label='Baseline', alpha=0.8, color='red')
    bars2 = ax1.bar(x, adversarial_mce, width, label='Adversarial Training', alpha=0.8, color='blue')
    bars3 = ax1.bar(x + width, aligned_mce, width, label='Human Aligned', alpha=0.8, color='green')
    
    ax1.set_ylabel('mCE (lower is better)')
    ax1.set_title('Mean Corruption Error (mCE) Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height != float('inf') and not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
    
    # Improvement comparison (relative to baseline)
    improvements = []
    categories = ['Adversarial Training', 'Human Aligned']
    
    for i, model in enumerate(models):
        baseline = baseline_mce[i]
        if baseline != float('inf') and not np.isnan(baseline) and baseline > 0:
            adv_improvement = (baseline - adversarial_mce[i]) / baseline * 100
            align_improvement = (baseline - aligned_mce[i]) / baseline * 100
            improvements.append([adv_improvement, align_improvement])
        else:
            improvements.append([0, 0])
    
    improvements = np.array(improvements)
    
    bars1 = ax2.bar(x - width/2, improvements[:, 0], width, label='Adversarial Training', alpha=0.8, color='blue')
    bars2 = ax2.bar(x + width/2, improvements[:, 1], width, label='Human Aligned', alpha=0.8, color='green')
    
    ax2.set_ylabel('Improvement over Baseline (%)')
    ax2.set_title('Robustness Improvement Relative to Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1 if height >= 0 else height - 3,
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_severity_analysis(all_results: dict, save_path: Path) -> None:
    """Plot accuracy vs corruption severity."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    models = list(all_results.keys())
    severities = [1, 2, 3, 4, 5]
    
    # Select a few representative corruptions
    selected_corruptions = ['gaussian_noise', 'motion_blur', 'brightness', 'contrast']
    
    for idx, corruption in enumerate(selected_corruptions):
        ax = axes[idx]
        
        for model in models:
            if corruption in all_results[model]:
                accuracies = all_results[model][corruption]['accuracies_by_severity']
                ax.plot(severities, accuracies, marker='o', label=model, linewidth=2)
        
        ax.set_xlabel('Corruption Severity')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{corruption.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(severities)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main distribution shift evaluation."""
    print("RobustSight: Distribution Shift Evaluation")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Models and training types to evaluate
    models_config = [
        ("ResNet18", create_resnet18),
        ("ViT-Small", create_vit_small)
    ]
    
    training_types = ["baseline", "adversarial", "aligned"]
    
    all_results = {}
    mce_results = {}
    
    # Reference clean error rates (approximate values for mCE computation)
    clean_errors = {
        'ResNet18': 5.0,
        'ViT-Small': 8.0
    }
    
    for model_name, model_fn in models_config:
        for training_type in training_types:
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name} ({training_type})")
            print(f"{'='*50}")
            
            try:
                # Load model
                model = load_model(model_name, model_fn, device, training_type)
                
                # Evaluate on all corruptions
                corruption_results = evaluate_all_corruptions(model, device)
                
                # Compute mCE
                mce = compute_mce(corruption_results, clean_errors[model_name])
                
                # Store results
                result_key = f"{model_name}_{training_type}"
                all_results[result_key] = corruption_results
                mce_results[result_key] = mce
                
                print(f"mCE: {mce:.2f}")
                
                # Save individual results
                result_path = results_dir / f"{result_key}_corruption_results.json"
                with open(result_path, 'w') as f:
                    json.dump({
                        'corruption_results': corruption_results,
                        'mce': mce,
                        'model_name': model_name,
                        'training_type': training_type
                    }, f, indent=2)
                    
            except Exception as e:
                print(f"Error evaluating {model_name} ({training_type}): {e}")
                result_key = f"{model_name}_{training_type}"
                all_results[result_key] = {}
                mce_results[result_key] = float('inf')
    
    # Create visualizations
    print(f"\n{'='*50}")
    print("Creating Visualizations")
    print(f"{'='*50}")
    
    # Corruption robustness heatmap
    heatmap_path = figures_dir / "corruption_robustness_heatmap.png"
    plot_corruption_robustness(all_results, heatmap_path)
    
    # mCE comparison
    mce_comparison_path = figures_dir / "mce_comparison.png"
    plot_mce_comparison(mce_results, mce_comparison_path)
    
    # Severity analysis
    severity_path = figures_dir / "corruption_severity_analysis.png"
    plot_severity_analysis(all_results, severity_path)
    
    # Save combined results
    combined_results = {
        'corruption_results': all_results,
        'mce_results': mce_results,
        'clean_error_rates': clean_errors
    }
    
    combined_path = results_dir / "distribution_shift_results.json"
    with open(combined_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("DISTRIBUTION SHIFT SUMMARY")
    print(f"{'='*50}")
    
    # Print mCE summary
    print("\nMean Corruption Error (mCE):")
    for model_name in ['ResNet18', 'ViT-Small']:
        print(f"\n{model_name}:")
        for training_type in training_types:
            key = f"{model_name}_{training_type}"
            mce = mce_results.get(key, float('inf'))
            if mce != float('inf'):
                print(f"  {training_type}: {mce:.2f}")
            else:
                print(f"  {training_type}: N/A")
    
    # Find best performing models
    print(f"\nBest Performing Models (by mCE):")
    valid_mce = {k: v for k, v in mce_results.items() if v != float('inf') and not np.isnan(v)}
    if valid_mce:
        best_model = min(valid_mce, key=valid_mce.get)
        print(f"Overall Best: {best_model} (mCE: {valid_mce[best_model]:.2f})")
    
    print("\nDistribution shift evaluation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
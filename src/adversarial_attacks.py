#!/usr/bin/env python3
"""
Adversarial attacks implementation: FGSM, PGD, and AutoAttack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from autoattack import AutoAttack
from train_baseline import create_resnet18, create_vit_small, get_data_loaders

class AdversarialAttacks:
    """Collection of adversarial attack methods."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def fgsm_attack(self, images: torch.Tensor, labels: torch.Tensor, 
                   epsilon: float) -> torch.Tensor:
        """Fast Gradient Sign Method (FGSM) attack."""
        images = images.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Collect datagrad
        data_grad = images.grad.data
        
        # Create adversarial examples
        sign_data_grad = data_grad.sign()
        perturbed_images = images + epsilon * sign_data_grad
        
        # Clamp to maintain [0,1] range after normalization
        perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)  # Adjusted for CIFAR-10 normalization
        
        return perturbed_images.detach()
    
    def pgd_attack(self, images: torch.Tensor, labels: torch.Tensor,
                   epsilon: float, alpha: float, num_iter: int) -> torch.Tensor:
        """Projected Gradient Descent (PGD) attack."""
        original_images = images.clone().detach()
        
        # Start with random noise
        perturbed_images = images.clone().detach()
        perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)
        
        for _ in range(num_iter):
            perturbed_images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images.detach() + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(perturbed_images - original_images, -epsilon, epsilon)
            perturbed_images = torch.clamp(original_images + delta, -2.5, 2.5)
        
        return perturbed_images.detach()
    
    def evaluate_attack(self, data_loader: DataLoader, attack_fn, **attack_params) -> dict:
        """Evaluate an attack on a dataset."""
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        all_clean_outputs = []
        all_adv_outputs = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Evaluating attack")):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Clean accuracy
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_pred = clean_outputs.max(1)[1]
                correct_clean += clean_pred.eq(labels).sum().item()
            
            # Generate adversarial examples
            if attack_fn == "autoattack":
                # AutoAttack is handled separately
                adv_images = images  # Placeholder
                adv_outputs = clean_outputs
            else:
                adv_images = attack_fn(images, labels, **attack_params)
                
                # Adversarial accuracy
                with torch.no_grad():
                    adv_outputs = self.model(adv_images)
            
            adv_pred = adv_outputs.max(1)[1]
            correct_adv += adv_pred.eq(labels).sum().item()
            
            total += labels.size(0)
            
            # Store outputs for analysis
            all_clean_outputs.append(clean_outputs.cpu())
            all_adv_outputs.append(adv_outputs.cpu())
            all_labels.append(labels.cpu())
            
            # Limit evaluation for speed during development
            if batch_idx >= 50:  # Evaluate on subset for faster testing
                break
        
        clean_acc = 100. * correct_clean / total
        adv_acc = 100. * correct_adv / total
        
        return {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'total_samples': total,
            'attack_success_rate': 100. * (correct_clean - correct_adv) / correct_clean if correct_clean > 0 else 0.0
        }

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

def run_autoattack(model: nn.Module, data_loader: DataLoader, 
                   device: torch.device, epsilon: float = 8/255) -> dict:
    """Run AutoAttack evaluation."""
    # Collect test data
    all_images = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        all_images.append(images)
        all_labels.append(labels)
        if batch_idx >= 10:  # Use subset for faster testing
            break
    
    test_images = torch.cat(all_images, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    
    # Initialize AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
    
    # Run attack
    print("Running AutoAttack (this may take a while)...")
    adv_examples = adversary.run_standard_evaluation(
        test_images.to(device), 
        test_labels.to(device), 
        bs=128
    )
    
    # Evaluate
    with torch.no_grad():
        clean_outputs = model(test_images.to(device))
        adv_outputs = model(adv_examples)
        
        clean_pred = clean_outputs.max(1)[1]
        adv_pred = adv_outputs.max(1)[1]
        
        clean_acc = clean_pred.eq(test_labels.to(device)).float().mean().item() * 100
        adv_acc = adv_pred.eq(test_labels.to(device)).float().mean().item() * 100
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'total_samples': len(test_labels),
        'attack_success_rate': 100. * (clean_acc - adv_acc) / clean_acc if clean_acc > 0 else 0.0
    }

def visualize_adversarial_examples(model: nn.Module, images: torch.Tensor, 
                                 labels: torch.Tensor, epsilon: float,
                                 save_path: Path, device: torch.device) -> None:
    """Visualize clean vs adversarial examples."""
    attacker = AdversarialAttacks(model, device)
    
    # Select first 8 images
    sample_images = images[:8].to(device)
    sample_labels = labels[:8].to(device)
    
    # Generate adversarial examples
    fgsm_adv = attacker.fgsm_attack(sample_images, sample_labels, epsilon)
    pgd_adv = attacker.pgd_attack(sample_images, sample_labels, epsilon, epsilon/4, 10)
    
    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    
    def denormalize(tensor):
        return torch.clamp(tensor * std + mean, 0, 1)
    
    sample_images_vis = denormalize(sample_images).cpu()
    fgsm_adv_vis = denormalize(fgsm_adv).cpu()
    pgd_adv_vis = denormalize(pgd_adv).cpu()
    
    # Create visualization
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(8):
        # Clean images
        axes[0, i].imshow(sample_images_vis[i].permute(1, 2, 0))
        axes[0, i].set_title(f'Clean\n{class_names[sample_labels[i]]}')
        axes[0, i].axis('off')
        
        # FGSM adversarial
        axes[1, i].imshow(fgsm_adv_vis[i].permute(1, 2, 0))
        axes[1, i].set_title(f'FGSM\nε={epsilon:.3f}')
        axes[1, i].axis('off')
        
        # PGD adversarial
        axes[2, i].imshow(pgd_adv_vis[i].permute(1, 2, 0))
        axes[2, i].set_title(f'PGD\nε={epsilon:.3f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attack_comparison(results: dict, save_path: Path) -> None:
    """Plot comparison of attack methods."""
    attacks = list(results.keys())
    models = list(results[attacks[0]].keys())
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Adversarial accuracy comparison
    x = np.arange(len(models))
    width = 0.25
    
    for i, attack in enumerate(attacks):
        adv_accs = [results[attack][model]['adversarial_accuracy'] for model in models]
        ax1.bar(x + i*width, adv_accs, width, label=attack)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Adversarial Accuracy (%)')
    ax1.set_title('Adversarial Accuracy by Attack Method')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Attack success rate comparison
    for i, attack in enumerate(attacks):
        success_rates = [results[attack][model]['attack_success_rate'] for model in models]
        ax2.bar(x + i*width, success_rates, width, label=attack)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Attack Success Rate (%)')
    ax2.set_title('Attack Success Rate by Method')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main adversarial attacks evaluation."""
    print("RobustSight: Adversarial Attacks Evaluation")
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
    _, test_loader = get_data_loaders()
    
    # Attack configurations
    epsilon = 8/255  # L_inf perturbation budget
    attack_configs = {
        'FGSM': {'epsilon': epsilon},
        'PGD': {'epsilon': epsilon, 'alpha': epsilon/4, 'num_iter': 10},
        # 'AutoAttack': {'epsilon': epsilon}  # Commented out for faster execution
    }
    
    all_results = {}
    
    for model_name, model_fn in models_config:
        print(f"\n{'='*50}")
        print(f"Evaluating attacks on {model_name}")
        print(f"{'='*50}")
        
        # Load model
        model = load_model(model_name, model_fn, device)
        attacker = AdversarialAttacks(model, device)
        
        model_results = {}
        
        for attack_name, attack_params in attack_configs.items():
            print(f"\nRunning {attack_name} attack...")
            
            if attack_name == 'FGSM':
                results = attacker.evaluate_attack(
                    test_loader, attacker.fgsm_attack, **attack_params
                )
            elif attack_name == 'PGD':
                results = attacker.evaluate_attack(
                    test_loader, attacker.pgd_attack, **attack_params
                )
            elif attack_name == 'AutoAttack':
                results = run_autoattack(model, test_loader, device, attack_params['epsilon'])
            
            model_results[attack_name] = results
            
            print(f"{attack_name} Results:")
            print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
            print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.2f}%")
            print(f"  Attack Success Rate: {results['attack_success_rate']:.2f}%")
        
        all_results[model_name] = model_results
        
        # Generate adversarial examples visualization
        sample_batch = next(iter(test_loader))
        vis_path = figures_dir / f"{model_name}_adversarial_examples.png"
        visualize_adversarial_examples(
            model, sample_batch[0], sample_batch[1], epsilon, vis_path, device
        )
    
    # Save results
    results_path = results_dir / "adversarial_attacks_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison plots
    # Reorganize results for plotting
    plot_results = {}
    for attack in attack_configs.keys():
        plot_results[attack] = {}
        for model in models_config:
            plot_results[attack][model[0]] = all_results[model[0]][attack]
    
    comparison_path = figures_dir / "adversarial_attacks_comparison.png"
    plot_attack_comparison(plot_results, comparison_path)
    
    print(f"\n{'='*50}")
    print("ADVERSARIAL ATTACKS SUMMARY")
    print(f"{'='*50}")
    
    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        for attack_name, attack_results in model_results.items():
            print(f"  {attack_name}:")
            print(f"    Adversarial Accuracy: {attack_results['adversarial_accuracy']:.2f}%")
            print(f"    Attack Success Rate: {attack_results['attack_success_rate']:.2f}%")
    
    print("\nAdversarial attacks evaluation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
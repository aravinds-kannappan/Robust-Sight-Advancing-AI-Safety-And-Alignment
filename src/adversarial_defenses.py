#!/usr/bin/env python3
"""
Adversarial defenses implementation: PGD-based adversarial training and randomized smoothing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from train_baseline import create_resnet18, create_vit_small, get_data_loaders
from adversarial_attacks import AdversarialAttacks

class AdversarialTrainer:
    """Adversarial training with PGD attacks."""
    
    def __init__(self, model: nn.Module, device: torch.device, epsilon: float = 8/255):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.alpha = epsilon / 4
        self.num_iter = 10
        
        self.train_losses = []
        self.train_clean_accs = []
        self.train_adv_accs = []
        self.val_clean_accs = []
        self.val_adv_accs = []
    
    def pgd_attack_training(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """PGD attack for training (inner maximization)."""
        original_images = images.clone().detach()
        
        # Start with random noise
        perturbed_images = images.clone().detach()
        perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-self.epsilon, self.epsilon)
        perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)
        
        for _ in range(self.num_iter):
            perturbed_images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images.detach() + self.alpha * data_grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(perturbed_images - original_images, -self.epsilon, self.epsilon)
            perturbed_images = torch.clamp(original_images + delta, -2.5, 2.5)
        
        return perturbed_images.detach()
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer, mix_ratio: float = 0.5) -> tuple[float, float, float]:
        """Train for one epoch with adversarial training."""
        self.model.train()
        running_loss = 0.0
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Adversarial Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Mixed batch: half clean, half adversarial
            clean_size = int(batch_size * mix_ratio)
            adv_size = batch_size - clean_size
            
            # Clean examples
            clean_inputs = inputs[:clean_size]
            clean_targets = targets[:clean_size]
            
            # Adversarial examples
            if adv_size > 0:
                adv_inputs_orig = inputs[clean_size:]
                adv_targets = targets[clean_size:]
                adv_inputs = self.pgd_attack_training(adv_inputs_orig, adv_targets)
                
                # Combine clean and adversarial
                mixed_inputs = torch.cat([clean_inputs, adv_inputs], dim=0)
                mixed_targets = torch.cat([clean_targets, adv_targets], dim=0)
            else:
                mixed_inputs = clean_inputs
                mixed_targets = clean_targets
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(mixed_inputs)
            loss = criterion(outputs, mixed_targets)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += mixed_targets.size(0)
            
            # Track clean vs adversarial accuracy separately
            if clean_size > 0:
                clean_correct += predicted[:clean_size].eq(clean_targets).sum().item()
            if adv_size > 0:
                adv_correct += predicted[clean_size:].eq(adv_targets).sum().item()
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Clean': f'{100.*clean_correct/(clean_size*(batch_idx+1)):.1f}%' if clean_size > 0 else 'N/A',
                    'Adv': f'{100.*adv_correct/(adv_size*(batch_idx+1)):.1f}%' if adv_size > 0 else 'N/A'
                })
        
        epoch_loss = running_loss / len(train_loader)
        clean_acc = 100. * clean_correct / (clean_size * len(train_loader)) if clean_size > 0 else 0
        adv_acc = 100. * adv_correct / (adv_size * len(train_loader)) if adv_size > 0 else 0
        
        return epoch_loss, clean_acc, adv_acc
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        """Validate with both clean and adversarial examples."""
        self.model.eval()
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        attacker = AdversarialAttacks(self.model, self.device)
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Clean accuracy
                clean_outputs = self.model(inputs)
                clean_pred = clean_outputs.max(1)[1]
                clean_correct += clean_pred.eq(targets).sum().item()
                
                # Adversarial accuracy
                adv_inputs = attacker.pgd_attack(inputs, targets, self.epsilon, self.alpha, self.num_iter)
                adv_outputs = self.model(adv_inputs)
                adv_pred = adv_outputs.max(1)[1]
                adv_correct += adv_pred.eq(targets).sum().item()
                
                total += targets.size(0)
        
        clean_acc = 100. * clean_correct / total
        adv_acc = 100. * adv_correct / total
        
        return clean_acc, adv_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, lr: float = 0.1) -> dict:
        """Full adversarial training loop."""
        print(f"Starting adversarial training for {epochs} epochs...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_adv_acc = 0
        start_time = datetime.now()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_clean_acc, train_adv_acc = self.train_epoch(
                train_loader, criterion, optimizer
            )
            self.train_losses.append(train_loss)
            self.train_clean_accs.append(train_clean_acc)
            self.train_adv_accs.append(train_adv_acc)
            
            # Validation
            val_clean_acc, val_adv_acc = self.validate(val_loader, criterion)
            self.val_clean_accs.append(val_clean_acc)
            self.val_adv_accs.append(val_adv_acc)
            
            # Learning rate update
            scheduler.step()
            
            # Save best model based on adversarial accuracy
            if val_adv_acc > best_adv_acc:
                best_adv_acc = val_adv_acc
                self.save_checkpoint(epoch, val_clean_acc, val_adv_acc)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Clean Acc: {train_clean_acc:.2f}%, Train Adv Acc: {train_adv_acc:.2f}%")
            print(f"Val Clean Acc: {val_clean_acc:.2f}%, Val Adv Acc: {val_adv_acc:.2f}%")
        
        training_time = datetime.now() - start_time
        print(f"\nAdversarial training completed in {training_time}")
        print(f"Best adversarial accuracy: {best_adv_acc:.2f}%")
        
        return {
            'training_time': str(training_time),
            'best_adv_acc': best_adv_acc,
            'train_losses': self.train_losses,
            'train_clean_accs': self.train_clean_accs,
            'train_adv_accs': self.train_adv_accs,
            'val_clean_accs': self.val_clean_accs,
            'val_adv_accs': self.val_adv_accs
        }
    
    def save_checkpoint(self, epoch: int, clean_acc: float, adv_acc: float) -> None:
        """Save model checkpoint."""
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_name = self.model.__class__.__name__
        checkpoint_path = models_dir / f"{model_name}_adversarial_best.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
        }, checkpoint_path)

class RandomizedSmoothing:
    """Randomized smoothing defense."""
    
    def __init__(self, model: nn.Module, device: torch.device, sigma: float = 0.25):
        self.model = model
        self.device = device
        self.sigma = sigma
    
    def predict_smooth(self, inputs: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """Predict with randomized smoothing."""
        self.model.eval()
        batch_size = inputs.size(0)
        num_classes = 10  # CIFAR-10
        
        # Collect votes
        votes = torch.zeros(batch_size, num_classes).to(self.device)
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Add Gaussian noise
                noise = torch.randn_like(inputs) * self.sigma
                noisy_inputs = inputs + noise
                
                # Get predictions
                outputs = self.model(noisy_inputs)
                predictions = F.softmax(outputs, dim=1)
                votes += predictions
        
        # Return averaged predictions
        return votes / num_samples
    
    def evaluate_smoothed(self, data_loader: DataLoader, num_samples: int = 100) -> dict:
        """Evaluate smoothed classifier."""
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(data_loader, desc="Randomized Smoothing Evaluation"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get smoothed predictions
            smooth_outputs = self.predict_smooth(inputs, num_samples)
            predictions = smooth_outputs.max(1)[1]
            
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return {
            'smoothed_accuracy': accuracy,
            'sigma': self.sigma,
            'num_samples': num_samples
        }

def plot_adversarial_training_curves(results: dict, model_name: str, save_path: Path) -> None:
    """Plot adversarial training curves."""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6))
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # Training loss
    ax1.plot(epochs, results['train_losses'], label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Adversarial Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracies
    ax2.plot(epochs, results['train_clean_accs'], label='Train Clean', color='blue')
    ax2.plot(epochs, results['train_adv_accs'], label='Train Adversarial', color='red')
    ax2.plot(epochs, results['val_clean_accs'], label='Val Clean', color='lightblue')
    ax2.plot(epochs, results['val_adv_accs'], label='Val Adversarial', color='lightcoral')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Adversarial Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_robustness_tradeoff(baseline_results: dict, adversarial_results: dict, 
                           save_path: Path) -> None:
    """Plot clean vs adversarial accuracy tradeoff."""
    models = list(baseline_results.keys())
    
    baseline_clean = [baseline_results[model]['clean_accuracy'] for model in models]
    baseline_adv = [0] * len(models)  # Assume 0 adversarial accuracy for baseline
    
    adv_clean = [adversarial_results[model]['val_clean_accs'][-1] for model in models]
    adv_robust = [adversarial_results[model]['val_adv_accs'][-1] for model in models]
    
    plt.figure(figsize=(10, 8))
    
    # Plot baseline models
    plt.scatter(baseline_clean, baseline_adv, s=100, c='red', marker='o', 
                label='Baseline Models', alpha=0.7)
    
    # Plot adversarially trained models
    plt.scatter(adv_clean, adv_robust, s=100, c='blue', marker='s', 
                label='Adversarially Trained', alpha=0.7)
    
    # Add model labels
    for i, model in enumerate(models):
        plt.annotate(f'{model}\n(Baseline)', (baseline_clean[i], baseline_adv[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
        plt.annotate(f'{model}\n(Adv. Trained)', (adv_clean[i], adv_robust[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Clean Accuracy (%)')
    plt.ylabel('Adversarial Accuracy (%)')
    plt.title('Clean vs Adversarial Accuracy Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main adversarial defenses training and evaluation."""
    print("RobustSight: Adversarial Defenses")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders()
    
    # Models to train with adversarial training
    models_config = [
        ("ResNet18", create_resnet18),
        ("ViT-Small", create_vit_small)
    ]
    
    adversarial_results = {}
    
    # Train adversarially robust models
    for model_name, model_fn in models_config:
        print(f"\n{'='*50}")
        print(f"Adversarial Training: {model_name}")
        print(f"{'='*50}")
        
        model = model_fn()
        trainer = AdversarialTrainer(model, device)
        
        # Train with adversarial training
        results = trainer.train(train_loader, test_loader, epochs=50)  # Reduced epochs for demo
        adversarial_results[model_name] = results
        
        # Plot training curves
        plot_path = figures_dir / f"{model_name}_adversarial_training.png"
        plot_adversarial_training_curves(results, model_name, plot_path)
        
        # Save results
        result_path = results_dir / f"{model_name}_adversarial_training.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Load baseline results for comparison
    baseline_path = results_dir / "baseline_evaluation_results.json"
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        
        # Plot robustness tradeoff
        tradeoff_path = figures_dir / "robustness_tradeoff.png"
        plot_robustness_tradeoff(baseline_results, adversarial_results, tradeoff_path)
    
    # Evaluate randomized smoothing
    print(f"\n{'='*50}")
    print("Randomized Smoothing Evaluation")
    print(f"{'='*50}")
    
    smoothing_results = {}
    
    for model_name, model_fn in models_config:
        print(f"\nEvaluating randomized smoothing: {model_name}")
        
        # Load baseline model
        models_dir = Path(__file__).parent.parent / "models"
        checkpoint_path = models_dir / f"{model_name}_best.pth"
        
        model = model_fn()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate with randomized smoothing
        smoother = RandomizedSmoothing(model, device, sigma=0.25)
        smooth_results = smoother.evaluate_smoothed(test_loader, num_samples=50)  # Reduced for demo
        
        smoothing_results[model_name] = smooth_results
        print(f"Smoothed accuracy: {smooth_results['smoothed_accuracy']:.2f}%")
    
    # Save all results
    all_defense_results = {
        'adversarial_training': adversarial_results,
        'randomized_smoothing': smoothing_results
    }
    
    defense_results_path = results_dir / "adversarial_defenses_results.json"
    with open(defense_results_path, 'w') as f:
        json.dump(all_defense_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("ADVERSARIAL DEFENSES SUMMARY")
    print(f"{'='*50}")
    
    for model_name in models_config:
        model_name = model_name[0]
        print(f"\n{model_name}:")
        
        # Adversarial training results
        adv_results = adversarial_results[model_name]
        print(f"  Adversarial Training:")
        print(f"    Final Clean Acc: {adv_results['val_clean_accs'][-1]:.2f}%")
        print(f"    Final Adv Acc: {adv_results['val_adv_accs'][-1]:.2f}%")
        
        # Randomized smoothing results
        smooth_results = smoothing_results[model_name]
        print(f"  Randomized Smoothing:")
        print(f"    Smoothed Acc: {smooth_results['smoothed_accuracy']:.2f}%")
    
    print("\nAdversarial defenses evaluation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
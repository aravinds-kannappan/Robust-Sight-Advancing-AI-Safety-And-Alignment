#!/usr/bin/env python3
"""
Baseline training for ResNet-18 and ViT models on CIFAR-10.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

class ModelTrainer:
    """Base trainer class for CIFAR-10 models."""
    
    def __init__(self, model_name: str, model: nn.Module, device: torch.device):
        self.model_name = model_name
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training {self.model_name}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validating {self.model_name}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        return val_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, lr: float = 0.1) -> dict:
        """Full training loop."""
        print(f"\nTraining {self.model_name} for {epochs} epochs...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0
        start_time = datetime.now()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_acc = self.validate(val_loader, criterion)
            self.val_accuracies.append(val_acc)
            
            # Learning rate update
            scheduler.step()
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        training_time = datetime.now() - start_time
        print(f"\nTraining completed in {training_time}")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        
        return {
            'model_name': self.model_name,
            'best_val_acc': best_acc,
            'training_time': str(training_time),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, epoch: int, accuracy: float) -> None:
        """Save model checkpoint."""
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        checkpoint_path = models_dir / f"{self.model_name}_best.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
        }, checkpoint_path)

def create_resnet18() -> nn.Module:
    """Create ResNet-18 model for CIFAR-10."""
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    
    # Modify first conv layer for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    
    return model

def create_vit_small() -> nn.Module:
    """Create ViT-Small model for CIFAR-10."""
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10, img_size=32)
    return model

def get_data_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 data loaders."""
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Get data directory
    data_dir = Path(__file__).parent.parent / "data" / "cifar10"
    
    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=False, transform=transform_test
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def plot_training_curves(results: dict, save_path: Path) -> None:
    """Plot and save training curves."""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    ax1.plot(results['train_losses'], label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{results["model_name"]} - Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2.plot(results['train_accuracies'], label='Training Accuracy', color='blue')
    ax2.plot(results['val_accuracies'], label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{results["model_name"]} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function."""
    print("RobustSight: Baseline Training")
    print("=" * 40)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders()
    
    # Models to train
    models_config = [
        ("ResNet18", create_resnet18()),
        ("ViT-Small", create_vit_small())
    ]
    
    all_results = {}
    
    for model_name, model in models_config:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        trainer = ModelTrainer(model_name, model, device)
        results = trainer.train(train_loader, test_loader, epochs=100)
        
        # Save training curves
        plot_path = figures_dir / f"{model_name}_training_curves.png"
        plot_training_curves(results, plot_path)
        
        all_results[model_name] = results
        
        # Save individual results
        result_path = results_dir / f"{model_name}_baseline_results.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save combined results
    combined_path = results_dir / "baseline_training_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("BASELINE TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for model_name, results in all_results.items():
        print(f"{model_name}: {results['best_val_acc']:.2f}% (training time: {results['training_time']})")
    
    print("\nBaseline training completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
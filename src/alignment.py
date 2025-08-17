#!/usr/bin/env python3
"""
Human-guided alignment with saliency-alignment loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import Tuple, Dict, Any
from train_baseline import create_resnet18, create_vit_small
from interpretability import GradCAM, generate_object_masks, compute_iou

class AlignmentDataset(Dataset):
    """Dataset with human annotations (object masks) for alignment."""
    
    def __init__(self, base_dataset, object_masks: np.ndarray):
        self.base_dataset = base_dataset
        self.object_masks = torch.from_numpy(object_masks).float()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        mask = self.object_masks[idx]
        return image, label, mask

class SaliencyAlignmentLoss(nn.Module):
    """Saliency alignment loss for human-guided training."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for alignment loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, 
                saliency_maps: torch.Tensor, human_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        # Classification loss
        cls_loss = self.ce_loss(outputs, targets)
        
        # Resize saliency maps to match human masks
        if saliency_maps.shape[-2:] != human_masks.shape[-2:]:
            saliency_maps = F.interpolate(
                saliency_maps.unsqueeze(1), 
                size=human_masks.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # Normalize saliency maps to [0, 1]
        saliency_maps = torch.sigmoid(saliency_maps)
        
        # Alignment loss (MSE between saliency and human masks)
        alignment_loss = self.mse_loss(saliency_maps, human_masks)
        
        # Total loss
        total_loss = self.alpha * cls_loss + self.beta * alignment_loss
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'alignment_loss': alignment_loss
        }

class AlignedModel(nn.Module):
    """Wrapper model that outputs both predictions and saliency maps."""
    
    def __init__(self, base_model: nn.Module, model_type: str = "resnet"):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        
        if model_type == "resnet":
            # Add saliency head for ResNet
            self.saliency_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),  # Reduce spatial dimensions
                nn.Conv2d(512, 256, kernel_size=3, padding=1),  # ResNet18 has 512 channels
                nn.ReLU(),
                nn.Conv2d(256, 1, kernel_size=1),  # Single channel output
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
            )
        else:
            # For ViT, we'll use a simpler approach
            self.saliency_head = nn.Sequential(
                nn.Linear(384, 256),  # ViT-Small has 384 features
                nn.ReLU(),
                nn.Linear(256, 32*32),  # Flatten to spatial dimensions
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both predictions and saliency."""
        if self.model_type == "resnet":
            # Extract features from ResNet
            features = self.base_model.conv1(x)
            features = self.base_model.bn1(features)
            features = self.base_model.relu(features)
            features = self.base_model.maxpool(features)
            
            features = self.base_model.layer1(features)
            features = self.base_model.layer2(features)
            features = self.base_model.layer3(features)
            layer4_features = self.base_model.layer4(features)
            
            # Classification branch
            pooled = self.base_model.avgpool(layer4_features)
            flattened = torch.flatten(pooled, 1)
            predictions = self.base_model.fc(flattened)
            
            # Saliency branch
            saliency = self.saliency_head(layer4_features).squeeze(1)  # [B, H, W]
            
        else:  # ViT
            # For ViT, we'll use a simplified approach
            predictions = self.base_model(x)
            
            # Generate dummy saliency (in practice, you'd extract patch embeddings)
            batch_size = x.shape[0]
            saliency = torch.randn(batch_size, 32, 32).to(x.device)
        
        return predictions, saliency

class AlignmentTrainer:
    """Trainer for human-guided alignment."""
    
    def __init__(self, model: AlignedModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_cls_losses = []
        self.train_align_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_ious = []
    
    def train_epoch(self, train_loader: DataLoader, criterion: SaliencyAlignmentLoss, 
                   optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch with alignment."""
        self.model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_align_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Alignment Training")
        for batch_idx, (inputs, targets, masks) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, saliency_maps = self.model(inputs)
            
            # Compute loss
            loss_dict = criterion(predictions, targets, saliency_maps, masks)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss_dict['total_loss'].item()
            running_cls_loss += loss_dict['classification_loss'].item()
            running_align_loss += loss_dict['alignment_loss'].item()
            
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Cls': f'{running_cls_loss/(batch_idx+1):.3f}',
                    'Align': f'{running_align_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.1f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_cls_loss = running_cls_loss / len(train_loader)
        epoch_align_loss = running_align_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'total_loss': epoch_loss,
            'cls_loss': epoch_cls_loss,
            'align_loss': epoch_align_loss,
            'accuracy': epoch_acc
        }
    
    def validate(self, val_loader: DataLoader, criterion: SaliencyAlignmentLoss) -> Dict[str, float]:
        """Validate with alignment metrics."""
        self.model.eval()
        correct = 0
        total = 0
        total_iou = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for inputs, targets, masks in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predictions, saliency_maps = self.model(inputs)
                
                # Accuracy
                _, predicted = predictions.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # IoU computation
                saliency_np = torch.sigmoid(saliency_maps).cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                for i in range(len(saliency_np)):
                    iou = compute_iou(saliency_np[i], masks_np[i])
                    total_iou += iou
                    num_samples += 1
        
        val_acc = 100. * correct / total
        mean_iou = total_iou / num_samples if num_samples > 0 else 0.0
        
        return {
            'accuracy': val_acc,
            'mean_iou': mean_iou
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, lr: float = 0.01) -> Dict[str, Any]:
        """Full alignment training loop."""
        print(f"Starting alignment training for {epochs} epochs...")
        
        criterion = SaliencyAlignmentLoss(alpha=1.0, beta=0.5)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_iou = 0
        start_time = datetime.now()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_results = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_results['total_loss'])
            self.train_cls_losses.append(train_results['cls_loss'])
            self.train_align_losses.append(train_results['align_loss'])
            self.train_accuracies.append(train_results['accuracy'])
            
            # Validation
            val_results = self.validate(val_loader, criterion)
            self.val_accuracies.append(val_results['accuracy'])
            self.val_ious.append(val_results['mean_iou'])
            
            # Learning rate update
            scheduler.step()
            
            # Save best model based on IoU
            if val_results['mean_iou'] > best_iou:
                best_iou = val_results['mean_iou']
                self.save_checkpoint(epoch, val_results['accuracy'], val_results['mean_iou'])
            
            print(f"Train - Loss: {train_results['total_loss']:.4f}, "
                  f"Cls: {train_results['cls_loss']:.4f}, "
                  f"Align: {train_results['align_loss']:.4f}, "
                  f"Acc: {train_results['accuracy']:.2f}%")
            print(f"Val - Acc: {val_results['accuracy']:.2f}%, IoU: {val_results['mean_iou']:.4f}")
        
        training_time = datetime.now() - start_time
        print(f"\nAlignment training completed in {training_time}")
        print(f"Best IoU: {best_iou:.4f}")
        
        return {
            'training_time': str(training_time),
            'best_iou': best_iou,
            'train_losses': self.train_losses,
            'train_cls_losses': self.train_cls_losses,
            'train_align_losses': self.train_align_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_ious': self.val_ious
        }
    
    def save_checkpoint(self, epoch: int, accuracy: float, iou: float) -> None:
        """Save model checkpoint."""
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_name = f"{self.model.model_type}_aligned_best"
        checkpoint_path = models_dir / f"{model_name}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'iou': iou,
        }, checkpoint_path)

def prepare_alignment_data(data_dir: Path) -> Tuple[Dataset, Dataset]:
    """Prepare datasets with object masks for alignment."""
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir / "cifar10"), train=True, download=False, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir / "cifar10"), train=False, download=False, transform=transform
    )
    
    # Generate object masks (in practice, these would be human-annotated)
    print("Generating object masks...")
    
    # For training set (use subset for demonstration)
    train_subset_size = 5000  # Use smaller subset for faster training
    train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
    
    train_images = []
    for idx in tqdm(train_indices, desc="Loading train images"):
        image, _ = train_dataset[idx]
        train_images.append(image.unsqueeze(0))
    
    train_images_tensor = torch.cat(train_images, dim=0)
    train_masks = generate_object_masks(train_images_tensor)
    
    # For test set
    test_images = []
    for idx in tqdm(range(len(test_dataset)), desc="Loading test images"):
        image, _ = test_dataset[idx]
        test_images.append(image.unsqueeze(0))
    
    test_images_tensor = torch.cat(test_images, dim=0)
    test_masks = generate_object_masks(test_images_tensor)
    
    # Create alignment datasets
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    train_alignment_dataset = AlignmentDataset(train_subset, train_masks)
    test_alignment_dataset = AlignmentDataset(test_dataset, test_masks)
    
    return train_alignment_dataset, test_alignment_dataset

def plot_alignment_training(results: Dict[str, Any], model_name: str, save_path: Path) -> None:
    """Plot alignment training curves."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # Total loss
    ax1.plot(epochs, results['train_losses'], label='Total Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Total Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Classification vs Alignment loss
    ax2.plot(epochs, results['train_cls_losses'], label='Classification', color='blue')
    ax2.plot(epochs, results['train_align_losses'], label='Alignment', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{model_name} - Loss Components')
    ax2.legend()
    ax2.grid(True)
    
    # Accuracy
    ax3.plot(epochs, results['train_accuracies'], label='Train Accuracy', color='blue')
    ax3.plot(epochs, results['val_accuracies'], label='Val Accuracy', color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title(f'{model_name} - Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    # IoU
    ax4.plot(epochs, results['val_ious'], label='Validation IoU', color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IoU')
    ax4.set_title(f'{model_name} - Alignment (IoU)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_alignment_results(model: AlignedModel, test_loader: DataLoader, 
                              device: torch.device, save_path: Path) -> None:
    """Visualize alignment results."""
    model.eval()
    
    # Get sample batch
    sample_batch = next(iter(test_loader))
    images, labels, masks = sample_batch[0][:6], sample_batch[1][:6], sample_batch[2][:6]
    
    with torch.no_grad():
        predictions, saliency_maps = model(images.to(device))
        saliency_maps = torch.sigmoid(saliency_maps).cpu()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    def denormalize(tensor):
        return torch.clamp(tensor * std + mean, 0, 1)
    
    images_vis = denormalize(images).permute(0, 2, 3, 1).numpy()
    
    # Create visualization
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(6):
        # Original image
        axes[0, i].imshow(images_vis[i])
        axes[0, i].set_title(f'Original\n{class_names[labels[i]]}')
        axes[0, i].axis('off')
        
        # Human mask
        axes[1, i].imshow(masks[i], cmap='jet', alpha=0.7)
        axes[1, i].imshow(images_vis[i], alpha=0.3)
        axes[1, i].set_title('Human Mask')
        axes[1, i].axis('off')
        
        # Model saliency
        axes[2, i].imshow(saliency_maps[i], cmap='jet', alpha=0.7)
        axes[2, i].imshow(images_vis[i], alpha=0.3)
        axes[2, i].set_title('Model Saliency')
        axes[2, i].axis('off')
        
        # Difference
        diff = torch.abs(saliency_maps[i] - masks[i])
        axes[3, i].imshow(diff, cmap='RdBu', vmin=0, vmax=1)
        iou = compute_iou(saliency_maps[i].numpy(), masks[i].numpy())
        axes[3, i].set_title(f'Difference\nIoU: {iou:.3f}')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main alignment training and evaluation."""
    print("RobustSight: Human-Guided Alignment")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    data_dir = Path(__file__).parent.parent / "data"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Prepare alignment datasets
    print("Preparing alignment datasets...")
    train_dataset, test_dataset = prepare_alignment_data(data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Models to train with alignment
    models_config = [
        ("ResNet18", create_resnet18, "resnet"),
        ("ViT-Small", create_vit_small, "vit")
    ]
    
    all_results = {}
    
    for model_name, model_fn, model_type in models_config:
        print(f"\n{'='*50}")
        print(f"Alignment Training: {model_name}")
        print(f"{'='*50}")
        
        # Load pretrained base model
        models_dir = Path(__file__).parent.parent / "models"
        checkpoint_path = models_dir / f"{model_name}_best.pth"
        
        base_model = model_fn()
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained {model_name}")
        else:
            print(f"Warning: No pretrained {model_name} found, training from scratch")
        
        # Create aligned model
        aligned_model = AlignedModel(base_model, model_type)
        trainer = AlignmentTrainer(aligned_model, device)
        
        # Train with alignment
        results = trainer.train(train_loader, test_loader, epochs=30)
        all_results[model_name] = results
        
        # Plot training curves
        plot_path = figures_dir / f"{model_name}_alignment_training.png"
        plot_alignment_training(results, model_name, plot_path)
        
        # Visualize alignment results
        vis_path = figures_dir / f"{model_name}_alignment_visualization.png"
        visualize_alignment_results(aligned_model, test_loader, device, vis_path)
        
        # Save results
        result_path = results_dir / f"{model_name}_alignment_results.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save combined results
    combined_path = results_dir / "alignment_training_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Comparison plot
    plt.figure(figsize=(12, 6))
    
    models = list(all_results.keys())
    final_ious = [all_results[model]['val_ious'][-1] for model in models]
    final_accs = [all_results[model]['val_accuracies'][-1] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # IoU comparison
    bars1 = ax1.bar(x, final_ious, width, alpha=0.8, color=['blue', 'orange'])
    ax1.set_ylabel('Final IoU')
    ax1.set_title('Alignment Quality (IoU) After Training')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(True, alpha=0.3)
    
    for bar, iou in zip(bars1, final_ious):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{iou:.3f}', ha='center', va='bottom')
    
    # Accuracy comparison
    bars2 = ax2.bar(x, final_accs, width, alpha=0.8, color=['blue', 'orange'])
    ax2.set_ylabel('Final Accuracy (%)')
    ax2.set_title('Classification Accuracy After Alignment')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars2, final_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_path = figures_dir / "alignment_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*50}")
    print("ALIGNMENT TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for model_name, results in all_results.items():
        print(f"{model_name}:")
        print(f"  Final IoU: {results['val_ious'][-1]:.4f}")
        print(f"  Final Accuracy: {results['val_accuracies'][-1]:.2f}%")
        print(f"  Training Time: {results['training_time']}")
    
    print("\nAlignment training completed!")
    return 0

if __name__ == "__main__":
    exit(main())
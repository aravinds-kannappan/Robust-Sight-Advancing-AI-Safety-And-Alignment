#!/usr/bin/env python3
"""
Download and prepare CIFAR-10 and CIFAR-10-C datasets for RobustSight project.
"""

import os
import requests
import tarfile
import numpy as np
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_tar_file(filepath: Path, extract_to: Path) -> None:
    """Extract tar file."""
    print(f"Extracting {filepath.name}...")
    with tarfile.open(filepath, 'r') as tar:
        tar.extractall(extract_to)

def download_cifar10(data_dir: Path) -> None:
    """Download CIFAR-10 dataset using torchvision."""
    print("Downloading CIFAR-10 dataset...")
    
    # Create data directories
    cifar10_dir = data_dir / "cifar10"
    cifar10_dir.mkdir(parents=True, exist_ok=True)
    
    # Download train and test sets
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(cifar10_dir), 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(cifar10_dir), 
        train=False, 
        download=True, 
        transform=transform
    )
    
    print(f"CIFAR-10 downloaded: {len(train_dataset)} train, {len(test_dataset)} test samples")

def download_cifar10c(data_dir: Path) -> None:
    """Download CIFAR-10-C dataset."""
    print("Downloading CIFAR-10-C dataset...")
    
    cifar10c_dir = data_dir / "cifar10c"
    cifar10c_dir.mkdir(parents=True, exist_ok=True)
    
    # CIFAR-10-C URLs from Zenodo
    urls = {
        "CIFAR-10-C.tar": "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1",
        "CIFAR-10-C-labels.npy": "https://zenodo.org/record/2535967/files/labels.npy?download=1"
    }
    
    for filename, url in urls.items():
        filepath = cifar10c_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {filename}...")
            download_file(url, filepath)
        else:
            print(f"{filename} already exists, skipping download")
    
    # Extract the tar file
    tar_path = cifar10c_dir / "CIFAR-10-C.tar"
    if tar_path.exists() and not (cifar10c_dir / "CIFAR-10-C").exists():
        extract_tar_file(tar_path, cifar10c_dir)
    
    print("CIFAR-10-C download completed")

def verify_datasets(data_dir: Path) -> bool:
    """Verify that datasets are properly downloaded."""
    print("Verifying datasets...")
    
    # Check CIFAR-10
    cifar10_path = data_dir / "cifar10" / "cifar-10-batches-py"
    if not cifar10_path.exists():
        print("ERROR: CIFAR-10 not found")
        return False
    
    # Check CIFAR-10-C
    cifar10c_path = data_dir / "cifar10c" / "CIFAR-10-C"
    if not cifar10c_path.exists():
        print("ERROR: CIFAR-10-C not found")
        return False
    
    # Check for corruption types
    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 
                   'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
                   'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
                   'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
    
    missing_corruptions = []
    for corruption in corruptions:
        corruption_file = cifar10c_path / f"{corruption}.npy"
        if not corruption_file.exists():
            missing_corruptions.append(corruption)
    
    if missing_corruptions:
        print(f"WARNING: Missing corruption types: {missing_corruptions}")
        return False
    
    print("All datasets verified successfully!")
    return True

def main():
    """Main function to download all datasets."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    
    print("RobustSight: Data Download Script")
    print("=" * 40)
    
    # Create data directory
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download datasets
        download_cifar10(data_dir)
        download_cifar10c(data_dir)
        
        # Verify downloads
        if verify_datasets(data_dir):
            print("\n✅ All datasets downloaded and verified successfully!")
        else:
            print("\n❌ Dataset verification failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
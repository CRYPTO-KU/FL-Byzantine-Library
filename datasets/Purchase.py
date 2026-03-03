"""
Dataset loader for Purchase100 dataset
This dataset contains 197,324 samples with 600 binary features and 100 classes.
Used in federated learning and privacy-preserving ML research.

Dataset statistics:
- Total samples: 197,324
- Features: 600 binary features
- Classes: 100 (class-imbalanced)
- Train samples: 160,000 (80 clients × 2,000 samples)
- Validation samples: 5,000
- Test samples: 5,000
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PurchaseDataset(Dataset):
    """
    Purchase100 dataset for 100-class classification.
    
    Each sample has 600 binary features and belongs to one of 100 classes.
    The dataset is class-imbalanced, commonly used in FL research.
    """
    
    def __init__(self, root='./data', train=True, download=False, transform=None, 
                 val_split=0.2, test_split=0.1, random_seed=42):
        """
        Args:
            root: Root directory containing the dataset
            train: If True, loads training data; False loads test data
            download: Not used (dataset must be manually placed)
            transform: Optional transform to apply to features
            val_split: Fraction for validation (default: 0.2)
            test_split: Fraction for test (default: 0.1)
            random_seed: Random seed for data split (default: 42)
        """
        self.root = Path(root) / 'purchase'
        self.train = train
        self.transform = transform
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Load and preprocess data
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        """
        Load and preprocess the Purchase100 data.
        
        The dataset file format is CSV with:
        - First column: class label (0-99)
        - Remaining 600 columns: binary features (0 or 1)
        
        Split: Train (70%), Val (20%), Test (10%)
        """
        # Load the dataset
        data_path = self.root / 'dataset_purchase'
        if not data_path.exists():
            raise FileNotFoundError(
                f"Purchase dataset not found at {data_path}. "
                f"Please extract dataset_purchase.tgz in {self.root}"
            )
        
        # Read CSV file (no header, first column is label, rest are features)
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
        
        # Separate labels and features
        labels = data[:, 0].astype(np.int64) - 1  # First column is class label (convert from 1-100 to 0-99)
        features = data[:, 1:]  # Remaining 600 columns are binary features
        
        # Create train/val/test split
        np.random.seed(self.random_seed)
        n_samples = len(features)
        indices = np.random.permutation(n_samples)
        
        # Calculate split sizes
        test_size = int(n_samples * self.test_split)
        val_size = int(n_samples * self.val_split)
        train_size = n_samples - test_size - val_size
        
        if self.train:
            # Training set
            selected_indices = indices[:train_size]
        else:
            # Test set (using the test portion)
            selected_indices = indices[train_size:]
        
        features = features[selected_indices]
        labels = labels[selected_indices]
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        
        return features, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: Tensor of shape (num_features,)
            label: Scalar tensor (0 or 1)
        """
        features = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    @property
    def num_features(self):
        """Return number of input features"""
        return self.data.shape[1]


def get_purchase_dataset(root='./data', download=False):
    """
    Get train and test datasets for Purchase100.
    
    Args:
        root: Root directory for dataset
        download: Whether to download (not implemented - manual download required)
    
    Returns:
        trainset, testset: Purchase dataset objects
    """
    trainset = PurchaseDataset(root=root, train=True, download=download, 
                              val_split=0.2, test_split=0.1)
    testset = PurchaseDataset(root=root, train=False, download=download, 
                             val_split=0.2, test_split=0.1)
    
    return trainset, testset


if __name__ == '__main__':
    # Test the dataset
    print("Testing Purchase100 Dataset...")
    
    # Create datasets
    train_dataset = PurchaseDataset(root='./data', train=True)
    test_dataset = PurchaseDataset(root='./data', train=False)
    
    print(f"\nTraining set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Check a sample
    sample, label = train_dataset[0]
    print(f"\nSample shape: {sample.shape}")
    print(f"Number of features: {sample.shape[0]}")
    print(f"Label: {label.item()}")
    print(f"Feature values (first 20): {sample[:20].numpy()}")
    
    # Check label distribution
    import numpy as np
    all_labels = train_dataset.targets.numpy()
    unique_labels = np.unique(all_labels)
    print(f"\nNumber of classes: {len(unique_labels)}")
    print(f"Label range: [{unique_labels.min()}, {unique_labels.max()}]")
    
    # Test data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Check a batch
    batch_features, batch_labels = next(iter(train_loader))
    print(f"\nBatch features shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch labels: {batch_labels[:10].numpy()}")
    
    print("\nDataset test completed successfully!")

"""
HAR (Human Activity Recognition) Dataset Loader
Dataset from UCI Machine Learning Repository: Human Activity Recognition Using Smartphones

Dataset statistics:
- Total samples: ~10,299 (7,352 train + 2,947 test)
- Features: 561 (accelerometer and gyroscope measurements)
- Classes: 6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# Activity label mapping
ACTIVITY_LABELS = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5
}


class HARDataset(Dataset):
    """
    HAR dataset for 6-class activity classification.
    
    Each sample has 561 features from accelerometer/gyroscope sensors.
    """
    
    def __init__(self, root='./data', train=True, download=False, transform=None):
        """
        Args:
            root: Root directory containing the HAR folder
            train: If True, loads training data; False loads test data
            download: Not used (dataset must be manually placed)
            transform: Optional transform to apply to features
        """
        self.root = Path(root) / 'HAR'
        self.train = train
        self.transform = transform
        
        # Load and preprocess data
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        """Load HAR data from CSV files."""
        filename = 'train.csv' if self.train else 'test.csv'
        data_path = self.root / filename
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"HAR dataset not found at {data_path}. "
                f"Please place train.csv and test.csv in {self.root}"
            )
        
        # Read CSV - last two columns are label ID and activity name
        df = pd.read_csv(data_path)
        
        # Extract features (all columns except last two)
        features = df.iloc[:, :-2].values.astype(np.float32)
        
        # Extract labels from the activity name column (last column)
        activity_names = df.iloc[:, -1].values
        labels = np.array([ACTIVITY_LABELS[name] for name in activity_names], dtype=np.int64)
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        
        return features, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: Tensor of shape (561,)
            label: Scalar tensor (0-5)
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
    
    @property
    def num_classes(self):
        """Return number of classes"""
        return len(ACTIVITY_LABELS)


def get_har_dataset(root='./data', download=False):
    """
    Get train and test datasets for HAR.
    
    Args:
        root: Root directory for dataset
        download: Not implemented
    
    Returns:
        trainset, testset: HAR dataset objects
    """
    trainset = HARDataset(root=root, train=True, download=download)
    testset = HARDataset(root=root, train=False, download=download)
    
    return trainset, testset


if __name__ == '__main__':
    # Test the dataset
    print("Testing HAR Dataset...")
    
    train_dataset = HARDataset(root='./data', train=True)
    test_dataset = HARDataset(root='./data', train=False)
    
    print(f"\nTraining set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    sample, label = train_dataset[0]
    print(f"\nSample shape: {sample.shape}")
    print(f"Number of features: {train_dataset.num_features}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Label: {label.item()}")
    
    # Check label distribution
    all_labels = train_dataset.targets.numpy()
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"\nLabel distribution:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  Class {lbl}: {cnt} samples")
    
    print("\nDataset test completed successfully!")

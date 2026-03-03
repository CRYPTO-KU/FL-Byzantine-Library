"""
Efficient neural network models for Purchase Prediction (Acquire Valued Shoppers Challenge)
Designed to be small and efficient for edge deployment while maintaining good performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PurchaseNet(nn.Module):
    """
    Fully connected network for purchase prediction (Purchase100 dataset).
    
    Architecture: Feedforward network with layers 600 -> 1024 -> 100.
    - Input layer: 600 binary features
    - Hidden layer: 1024 units with ReLU activation and dropout
    - Output layer: 100 classes
    """
    
    def __init__(self, num_features=600, num_classes=100, hidden_dim=1024, dropout=0.3):
        """
        Args:
            num_features: Number of input features (default: 600)
            num_classes: Number of output classes (default: 100)
            hidden_dim: Hidden layer dimension (default: 1024)
            dropout: Dropout rate (default: 0.3)
        """
        super(PurchaseNet, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Hidden layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model with Purchase100 dataset specs
    batch_size = 32
    num_features = 600
    num_classes = 100
    
    # Test PurchaseNet
    model = PurchaseNet(num_features=num_features, num_classes=num_classes)
    x = torch.randn(batch_size, num_features)
    output = model(x)
    print(f"PurchaseNet (Purchase100):")
    print(f"  Architecture: {num_features} -> 1024 -> {num_classes}")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {count_parameters(model):,}")

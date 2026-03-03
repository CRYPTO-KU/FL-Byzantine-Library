"""
HAR (Human Activity Recognition) Models
Simple logistic regression for 6-class activity classification.
"""

import torch
import torch.nn as nn


class HARLogReg(nn.Module):
    """
    Logistic Regression for HAR dataset.
    
    Simple linear model: 561 features -> 6 classes
    """
    
    def __init__(self, num_features=561, num_classes=6):
        """
        Args:
            num_features: Number of input features (default: 561)
            num_classes: Number of output classes (default: 6)
        """
        super(HARLogReg, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.fc(x)


class HARMLP(nn.Module):
    """
    Simple MLP for HAR dataset.
    
    Architecture: 561 -> 256 -> 128 -> 6
    """
    
    def __init__(self, num_features=561, num_classes=6, hidden_dims=(256, 128), dropout=0.3):
        super(HARMLP, self).__init__()
        
        layers = []
        in_dim = num_features
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    # Test models
    batch_size = 32
    num_features = 561
    num_classes = 6
    
    x = torch.randn(batch_size, num_features)
    
    # Test LogReg
    logreg = HARLogReg(num_features, num_classes)
    out = logreg(x)
    print(f"HARLogReg: {x.shape} -> {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in logreg.parameters())}")
    
    # Test MLP
    mlp = HARMLP(num_features, num_classes)
    out = mlp(x)
    print(f"HARMLP: {x.shape} -> {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mlp.parameters())}")

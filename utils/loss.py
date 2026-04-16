import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean',ignore_index=255):
        """
        Focal Loss implementation with support for class weights
        
        Args:
            weight (torch.Tensor): Class weights tensor of shape (C,) where C is number of classes
            gamma (float): Focusing parameter. Higher gamma gives more weight to hard examples
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        
    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): Model predictions of shape (N, C, H, W) for semantic segmentation
            target (torch.Tensor): Ground truth labels of shape (N, H, W)
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Convert target to one-hot encoding
        ce_loss = F.cross_entropy(
            input, 
            target,
            weight=self.weight,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        pt = torch.exp(-ce_loss)  # Get the prediction probability
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
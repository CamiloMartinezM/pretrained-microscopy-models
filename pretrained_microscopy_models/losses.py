import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=None, size_average=True, weights=None):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.weights = weights
        self.__name__ = "DiceBCELoss"

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, weight=self.weights, reduction="mean")
        Dice_BCE = self.bce_weight * BCE + (1 - self.bce_weight) * dice_loss

        return Dice_BCE

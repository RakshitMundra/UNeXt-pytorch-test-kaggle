import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalTverskyLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for sparse / thin-structure binary segmentation.

    The Tversky index generalizes Dice with separate weights on false positives
    (alpha) and false negatives (beta). For roads (~few % positive pixels) we set
    beta > alpha so missed road pixels (FN) are penalized harder than spurious
    ones (FP), which pushes recall up on thin structures. Raising (1 - TI) to the
    power gamma (the "focal" term, exponent 0.75 == gamma 4/3 from Abraham &
    Khan, 2019) concentrates gradient on the hard, low-overlap images instead of
    the already-easy ones. Operates on raw logits, mirroring BCEDiceLoss.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        tp = (input * target).sum(1)
        fp = (input * (1 - target)).sum(1)
        fn = ((1 - input) * target).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).pow(self.gamma).mean()


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

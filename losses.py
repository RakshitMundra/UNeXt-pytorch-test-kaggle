import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalTverskyLoss',
           'FocalLoss', 'TverskyLoss', 'FocalPlusTverskyLoss']


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
    ones (FP), which pushes recall up on thin structures -- but pushing it too far
    over-segments and *lowers* IoU@0.5, so keep the asymmetry mild (e.g. 0.4/0.6).
    The "focal" term raises (1 - TI) to the power gamma to concentrate gradient on
    the hard, low-overlap images: this only down-weights easy examples when
    gamma > 1 (Abraham & Khan 2019 recommend gamma = 4/3). gamma = 1 disables the
    focal term (plain Tversky). gamma < 1 is *anti*-focal -- don't use it.
    Operates on raw logits, mirroring BCEDiceLoss.
    """

    def __init__(self, alpha=0.4, beta=0.6, gamma=1.0, smooth=1e-5):
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


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017).

    Standard BCE re-weighted by (1 - p_t)^gamma so well-classified (easy) pixels
    contribute little and the model focuses on hard ones -- this REQUIRES
    gamma > 1 to suppress easy examples (gamma = 0 reduces to plain weighted BCE).
    alpha is the weight on the positive (rare road) class.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p = torch.sigmoid(input)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss (1 - Tversky index). alpha weights FP, beta weights FN;
    beta > alpha favors recall. alpha = beta = 0.5 reduces to Dice."""

    def __init__(self, alpha=0.4, beta=0.6, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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
        return (1 - tversky).mean()


class FocalPlusTverskyLoss(nn.Module):
    """Weighted SUM of binary Focal loss and Tversky loss:

        loss = focal_weight * FocalLoss + tversky_weight * TverskyLoss

    This is NOT the FocalTverskyLoss (1 - TI)^gamma formulation -- it adds two
    separate terms. Focal handles per-pixel hard-example mining; Tversky handles
    region overlap / FP-FN balance. Operates on raw logits.
    """

    def __init__(self, focal_weight=1.0, tversky_weight=1.0,
                 focal_alpha=0.25, focal_gamma=2.0,
                 tversky_alpha=0.4, tversky_beta=0.6, smooth=1e-5):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)

    def forward(self, input, target):
        return (self.focal_weight * self.focal(input, target)
                + self.tversky_weight * self.tversky(input, target))


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

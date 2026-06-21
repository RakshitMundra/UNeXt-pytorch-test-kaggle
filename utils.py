import argparse
import torch
import torch.nn as nn

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


# ImageNet stats (albumentations Normalize defaults), kept on CPU and moved to
# the input's device per batch. normalize_on_gpu reproduces the previous CPU
# pipeline exactly: ((x/255 - mean)/std)/255, where x is uint8 [0,255] CHW from
# the Dataset. NOTE: the trailing /255 is legacy double-scaling that was present
# in the original dataset.py (after albumentations Normalize); it is preserved
# here so training/eval results are unchanged.
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_on_gpu(input, target):
    input = input.cuda(non_blocking=True).float()
    mean = _NORM_MEAN.to(input.device)
    std = _NORM_STD.to(input.device)
    input = (input / 255.0 - mean) / std
    input = input / 255.0
    target = target.cuda(non_blocking=True).float() / 255.0
    return input, target

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

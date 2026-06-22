import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations import HorizontalFlip, VerticalFlip
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool, normalize_on_gpu
from archs import UNext
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

# change for kaggle
DATA_ROOT = os.environ.get('DATA_ROOT', 'inputs')




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--resume', default=False, type=str2bool,
                    help='resume training from models/<name>/last_ckpt.pth')
    parser.add_argument('--init_weights', default='', type=str,
                    help='warm-start: load ONLY model weights from this checkpoint '
                         '(.pth, dict or bare state_dict) into a fresh run with a '
                         'fresh optimizer/scheduler. Use this when changing the loss '
                         'so the new objective is not held back by stale Adam moments '
                         'or a decayed cosine LR. Ignored when --resume is True.')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache', default=False, type=str2bool,
                        help='decode the whole (pre-resized) dataset into RAM once '
                             'and share it across workers via copy-on-write')

    config = parser.parse_args()

    return config

# args = parser.parse_args()
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input, target = normalize_on_gpu(input, target)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input, target = normalize_on_gpu(input, target)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Warm-start from old weights without inheriting optimizer/scheduler state.
    # Done before the (optional) DataParallel wrap so the keys match the bare model.
    if config.get('init_weights') and not config.get('resume'):
        ckpt = torch.load(config['init_weights'], map_location='cuda', weights_only=False)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state)
        print('=> warm-started model weights from %s (fresh optimizer + scheduler)'
              % config['init_weights'])

    if torch.cuda.device_count() > 1:
        print(f"=> using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # below changed for kaggle env
    img_ids = glob(os.path.join(DATA_ROOT, config['dataset'], '*_sat' + config['img_ext']))
    img_ids = [os.path.basename(p)[:-len('_sat' + config['img_ext'])] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Resize(config['input_h'], config['input_w']),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(DATA_ROOT, config['dataset']),
        mask_dir=os.path.join(DATA_ROOT, config['dataset']),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform,
        cache=config['cache'])
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(DATA_ROOT, config['dataset']),
        mask_dir=os.path.join(DATA_ROOT, config['dataset']),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform,
        cache=config['cache'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=config['num_workers'] > 0,
        prefetch_factor=4 if config['num_workers'] > 0 else None,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=config['num_workers'] > 0,
        prefetch_factor=4 if config['num_workers'] > 0 else None,
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    start_epoch = 0

    if config.get('resume'):
        # weights_only=False: our own trusted checkpoint, and it holds a python/
        # numpy best_iou scalar that the PyTorch 2.6+ safe loader rejects by default.
        ckpt = torch.load('models/%s/checkpoint.pth' % config['name'], weights_only=False)
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_iou = ckpt['best_iou']
        print('=> resumed from epoch %d (best_iou %.4f)' % (start_epoch, best_iou))
    
    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        # --- live loss/IoU curves ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(log['epoch'], log['loss'], label='train')
        ax[0].plot(log['epoch'], log['val_loss'], label='val')
        ax[0].set_title('Loss'); ax[0].legend()
        ax[1].plot(log['epoch'], log['iou'], label='train')
        ax[1].plot(log['epoch'], log['val_iou'], label='val')
        ax[1].set_title('IoU'); ax[1].legend()
        # fig.savefig('models/%s/curves.png' % config['name'])
        plt.close(fig)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save({
                'epoch': epoch,
                'best_iou': float(val_log['iou']),
                'model': (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }, 'models/%s/checkpoint.pth' % config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    # Final epoch save once loop ends
    torch.save((model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
               'models/%s/model_last.pth' % config['name'])


if __name__ == '__main__':
    main()

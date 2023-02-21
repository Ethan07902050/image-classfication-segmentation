import os
import copy
import time
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from net import Segnet
from dataset import SegmentationDataset
from fcn32s import FCN32s
from loss import DiceLoss, FocalLoss, FocalDiceLoss

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights

torch.manual_seed(42)


def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6

    return mean_iou


def create_dataloader(image_dir, phase, batch_size, crop_resize=False):
    data = SegmentationDataset(image_dir, phase=phase, crop_resize=crop_resize)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


def validate(dataloader, model, criterion, device):
    running_loss = 0.0
    all_preds, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if type(outputs) is dict:
                losses = {
                    key: criterion(item, labels)
                    for key, item in outputs.items()
                }
                loss = losses['out'] + 0.5 * losses['aux']
                outputs = outputs['out']
            else:
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            labels = labels.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            all_labels.append(labels)
            all_preds.append(preds)

            running_loss += loss.item() * inputs.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    val_iou = mean_iou_score(preds, labels)
    return val_loss, val_iou


def train(
    save_path,
    dataloaders,
    model,
    device,
    criterion,
    args
): 
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0.0

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) 
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()

    if args.scheduler == 'linear':
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.num_epochs, verbose=True)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.8, verbose=True)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, verbose=True)
    log = open(save_path / 'log.txt', 'w') 

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
        print('-' * 10)
        log.write(f'Epoch {epoch + 1}\n')

        running_loss = 0.0

        # Iterate over data.
        total = len(dataloaders['train'])
        for idx, (inputs, labels) in tqdm(enumerate(dataloaders['train']), total=total):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            # Output of fcn-resnet101, fcn-resnet50, deeplabv3-resnet50, deeplabv3-mobilenet
            if type(outputs) is dict:
                losses = {
                    key: criterion(item, labels)
                    for key, item in outputs.items()
                }
                loss = losses['out'] + 0.5 * losses['aux']
                outputs = outputs['out']
            else:
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            loss = loss / args.accum_step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            if (idx + 1) % args.accum_step == 0 or idx + 1 == len(dataloaders['train']):
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % args.validate_step == 0 or idx + 1 == len(dataloaders['train']):
                if (idx + 1) % args.validate_step == 0:
                    train_loss = running_loss / (args.validate_step * args.batch_size)
                else:
                    train_loss = running_loss / ((len(dataloaders['train']) % args.validate_step) * args.batch_size)
                val_loss, val_iou = validate(dataloaders['val'], model, criterion, device)

                print(f'train loss: {train_loss:.4f}')
                print(f'val loss: {val_loss:.4f}, val iou: {val_iou:.4f}')
                log.write(f'step: {idx+1}\n')
                log.write(f'train loss: {train_loss:.4f}\n')
                log.write(f'val loss: {val_loss:.4f}, val iou: {val_iou:.4f}\n')

                running_loss = 0.0
                if val_iou > best_iou:
                    best_iou = val_iou
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            filename = f'epoch-{epoch+1}.pt'
            torch.save(model.state_dict(), save_path / filename)

    filename = 'best.pt'
    torch.save(best_model_wts, save_path / filename)
    log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='')
    parser.add_argument('--val-dir', type=str, default='')
    parser.add_argument('--model-name', type=str, choices=['fcn32', 'fcn-resnet101', 'fcn-resnet50', 'deeplabv3-resnet50', 'deeplabv3-mobilenet'], default='fcn32')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=2e-5)
    parser.add_argument('--scheduler', type=str, choices=['linear', 'cosine', 'step', 'plateau'], default='linear')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--opt', type=str, choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--loss', type=str, choices=['cross-entropy', 'dice', 'focal-dice'], default='cross-entropy')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accum-step', type=int, default=16)
    parser.add_argument('--validate-step', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='ckpt/p2')
    parser.add_argument('--crop-resize', action='store_true')
    args = parser.parse_args()

    save_path = Path(args.output_dir) / time.strftime('%Y_%m_%d-%H_%M_%S')
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'config', 'w') as f:
        json.dump(vars(args), f, indent=2)

    dataloaders = {
        'train': create_dataloader(Path(args.train_dir), 'train', args.batch_size, args.crop_resize),
        'val': create_dataloader(Path(args.val_dir), 'val', args.batch_size),
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.loss == 'cross-entropy':
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == 'dice':
        criterion = DiceLoss('multiclass')
    elif args.loss == 'focal-dice':
        criterion = FocalDiceLoss()

    if args.model_name == 'fcn32':
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights)
        model = FCN32s(vgg)
    else:
        model = Segnet(args.model_name)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    if args.train:
        train(
            save_path=save_path, 
            dataloaders=dataloaders,
            model=model,
            device=device,
            criterion=criterion,
            args=args
        )
    if args.validate:
        loss, iou = validate(dataloaders['val'], model, criterion, device)
        print(f'iou: {iou}')

if __name__ == '__main__':
    main()
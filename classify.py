import os
import copy
import time
import json
import argparse
import numpy as np
import plotly.express as px
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from net import Net
from dataset import ClassificationDataset

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, StepLR
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)


def create_dataloader(image_dir, phase, batch_size, model_name, augment):
    data = ClassificationDataset(image_dir, phase, model_name, augment)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


def plot(method, save_path, dataloader, model, device):
    features, y = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            feature = model.net(inputs)
            feature = torch.flatten(feature, start_dim=1)
            features.append(feature.cpu().detach().numpy())
            y.append(labels.cpu().detach().numpy())

    features = np.concatenate(features, axis=0)
    y = np.concatenate(y, axis=0)

    if method == 'pca':
        pca = PCA(n_components=2)
        results = pca.fit_transform(features)
    elif method == 'tsne':
        tsne = TSNE(n_components=2, verbose=1)
        results = tsne.fit_transform(features)
    
    fig = px.scatter(results, x=0, y=1, color=y)
    fig.write_image(save_path / f'{method}.png')


def validate(dataloader, model, criterion, device):
    running_loss = 0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)   

    val_loss = running_loss / total
    val_acc = running_corrects / total
    return val_loss, val_acc


def train(
    save_path,
    dataloaders,
    model,
    device,
    criterion,
    args
): 
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)

    if args.scheduler == 'linear':
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.num_epochs, verbose=True)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=1, gamma=0.8, verbose=True)
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
        running_corrects = 0

        # Iterate over data.
        total = len(dataloaders['train'])
        for idx, (inputs, labels) in tqdm(enumerate(dataloaders['train']), total=total):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            loss = loss / args.accum_step
            loss.backward()

            if (idx + 1) % args.accum_step == 0 or idx + 1 == len(dataloaders['train']):
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % args.validate_step == 0 or idx + 1 == len(dataloaders['train']):
                if idx + 1 == len(dataloaders['train']):
                    train_loss = running_loss / ((len(dataloaders['train']) % args.validate_step) * args.batch_size)
                    train_acc = running_corrects / ((len(dataloaders['train']) % args.validate_step) * args.batch_size)
                else:
                    train_loss = running_loss / (args.validate_step * args.batch_size)
                    train_acc = running_corrects / (args.validate_step * args.batch_size)
                val_loss, val_acc = validate(dataloaders['val'], model, criterion, device)

                print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
                print(f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')
                log.write(f'step: {idx+1}\n')
                log.write(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}\n')
                log.write(f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}\n')

                running_loss = 0.0
                running_corrects = 0

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()
        scheduler.step()
        filename = f'epoch-{epoch+1}.pt'
        torch.save(model.state_dict(), save_path / filename)

    filename = 'best.pt'
    torch.save(best_model_wts, save_path / filename)
    log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='')
    parser.add_argument('--val-dir', type=str, default='')
    parser.add_argument('--model-name', type=str, choices=['resnet', 'efficientnet', 'densenet', 'convnext'], default='resnet')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--plot-method', type=str, default='')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--opt', type=str, choices=['sgd', 'rmsprop'], default='sgd')
    parser.add_argument('--weight-decay', type=float, default=2e-5)
    parser.add_argument('--scheduler', type=str, choices=['linear', 'cosine', 'step', 'plateau'], default='linear')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accum-step', type=int, default=16)
    parser.add_argument('--validate-step', type=int, default=500)
    parser.add_argument('--augment', type=str, choices=['random', 'auto'], default='random')
    parser.add_argument('--output-dir', type=str, default='ckpt/p1')
    args = parser.parse_args()

    save_path = Path(args.output_dir) / time.strftime('%Y_%m_%d-%H_%M_%S')
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'config', 'w') as f:
        json.dump(vars(args), f, indent=2)

    dataloaders = {
        'train': create_dataloader(Path(args.train_dir), 'train', args.batch_size, args.model_name, args.augment),
        'val': create_dataloader(Path(args.val_dir), 'val', args.batch_size, args.model_name, args.augment),
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Net(args.model_name, pretrained=args.pretrained)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
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
        validate(dataloaders['val'], model, criterion, device)
    if args.plot_method:
        plot(args.plot_method, save_path, dataloaders['val'], model, device)


if __name__ == '__main__':
    main()
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from net import Net
from dataset import ClassificationDataset

def predict(output_path, dataloader, model, device):
    f = open(output_path, 'w')
    f.write('filename,label\n')

    with torch.no_grad():
        for inputs, filename in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            f.write(f'{filename[0]},{preds.item()}\n')

    f.close()
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

    data = ClassificationDataset(Path(args.input_dir), phase='test', model_name='convnext')
    dataloader = DataLoader(data, batch_size=1)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net('convnext')
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    predict(args.output_path, dataloader, model, device)

if __name__ == '__main__':
    main()
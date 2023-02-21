import argparse
import imageio
import numpy as np
import copy
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from net import Segnet
from dataset import SegmentationDataset

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

def predict(model_paths, dataloader, output_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Segnet('fcn-resnet101')
    models = []

    for path in model_paths:
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        models.append(copy.deepcopy(model))

    with torch.no_grad():
        for img, filename in tqdm(dataloader):
            img = img.to(device)
            outputs = np.empty((len(model_paths), 7, 512, 512))
   
            for i, path in enumerate(model_paths):
                output = models[i](img)['out']
                output = output.cpu().detach().numpy().squeeze()
                outputs[i] = output

            preds = np.mean(outputs, axis=0)
            preds = np.argmax(preds, axis=0)

            mask = np.zeros((512, 512, 3))
            for cl, color in cls_color.items():
                mask[preds == cl] = color      

            imageio.imwrite(output_dir / f'{filename[0]}.png', np.uint8(mask))  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-paths', nargs='+', default=[])
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
        
    dataset = SegmentationDataset(Path(args.input_dir), phase='test')
    dataloader = DataLoader(dataset, batch_size=1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predict(args.model_paths, dataloader, output_path)

if __name__ == '__main__':
    main()
from PIL import Image

from torchvision.models import EfficientNet_V2_M_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models import VGG16_Weights
import torchvision.transforms.functional as F
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio


class ClassificationDataset(Dataset):
    def __init__(self, img_dir, phase='train', model_name='resnet', augment='random'):
        self.files = list(img_dir.glob('**/*'))
        self.phase = phase
        self.augment = augment
        if model_name == 'efficientnet':
            self.transform = EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
        elif model_name == 'densenet':
            self.transform = DenseNet121_Weights.IMAGENET1K_V1.transforms()
        elif model_name == 'convnext':
            self.transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        elif model_name == 'resnet':
            self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        if self.phase == 'train':
            if self.augment == 'auto':
                policy = transforms.AutoAugmentPolicy.CIFAR10
                augmenter = transforms.AutoAugment(policy)
            elif self.augment == 'random':
                augmenter = transforms.RandAugment()
            else:
                raise RuntimeError(f'Invalid augment method {self.augment}')
            image = augmenter(image)

        image = self.transform(image)
        if self.phase == 'train':
            eraser = transforms.RandomErasing()
            image = eraser(image)

        if self.phase == 'test':
            return image, self.files[idx].name

        label = int(self.files[idx].stem.split('_')[0])
        return image, label


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, phase='train', crop_resize=False):
        self.images = sorted(img_dir.glob('**/*.jpg'))
        self.phase = phase
        if self.phase != 'test':
            self.masks = sorted(img_dir.glob('**/*.png'))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.vertical_flip = transforms.RandomVerticalFlip(p=1)
        self.crop_resize = crop_resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])

        if self.phase != 'test':
            mask = Image.open(self.masks[idx])

        if self.phase == 'train':
            p = np.random.rand(3)
            if p[0] >= 0.5:
                image = self.horizontal_flip(image)
                mask = self.horizontal_flip(mask)
            if p[1] >= 0.5:
                image = self.vertical_flip(image)
                mask = self.vertical_flip(mask)

            if self.crop_resize and p[2] >= 0.5:
                image, mask = self.random_resized_crop(image, mask)

        image = self.transform(image)

        if self.phase != 'test':
            mask = np.array(mask)
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            label = np.zeros((512, 512))
            label[mask == 3] = 0  # (Cyan: 011) Urban land 
            label[mask == 6] = 1  # (Yellow: 110) Agriculture land 
            label[mask == 5] = 2  # (Purple: 101) Rangeland 
            label[mask == 2] = 3  # (Green: 010) Forest land 
            label[mask == 1] = 4  # (Blue: 001) Water 
            label[mask == 7] = 5  # (White: 111) Barren land 
            label[mask == 0] = 6  # (Black: 000) Unknown 
            label = torch.as_tensor(label, dtype=torch.int64)
            return image, label

        return image, self.images[idx].stem

    def random_resized_crop(self, img, mask):
        channels, height, width = F.get_dimensions(img)
        top = np.random.randint(low=0, high=round(height * 0.25))
        left = np.random.randint(low=0, high=round(width * 0.25))
        bottom = np.random.randint(low=round(height * 0.75), high=height)
        right = np.random.randint(low=round(width * 0.75), high=width)
        h = bottom - top
        w = right - left

        img = F.resized_crop(img, top, left, h, w, size=(512, 512))
        mask = F.resized_crop(mask, top, left, h, w, size=(512, 512))

        return img, mask

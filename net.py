import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

class Net(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        if model_name == 'convnext':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = convnext_tiny(weights=weights)
        elif model_name == 'densenet':
            weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = densenet121(weights=weights)
        elif model_name == 'resnet':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.net = resnet50(weights=weights)
        elif model_name == 'efficientnet':
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = efficientnet_v2_m(weights=weights)
        else:
            raise RuntimeError(f'unknown model name {model_name}')
        self.fc = nn.Linear(1000, 50)

    def forward(self, x):
        x = self.net(x)
        return self.fc(x)


class Segnet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == 'fcn-resnet101':
            weights = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            self.net = fcn_resnet101(weights=weights)
        elif model_name == 'fcn-resnet50':
            weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            self.net = fcn_resnet50(weights=weights)
        elif model_name == 'deeplabv3-resnet50':
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            self.net = deeplabv3_resnet50(weights=weights)
        elif model_name == 'deeplabv3-mobilenet':
            weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self.net = deeplabv3_mobilenet_v3_large(weights=weights)

        self.conv_out = nn.Conv2d(21, 7, 1, padding=0)
        self.conv_aux = nn.Conv2d(21, 7, 1, padding=0)

    def forward(self, x):
        x = self.net(x)
        return {
            'out': self.conv_out(x['out']),
            'aux': self.conv_aux(x['aux'])
        }
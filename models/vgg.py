# -*- coding: utf-8 -*-
# Author  : liyanpeng
# Email   : yanpeng.li@cumt.edu.cn
# Datetime: 2024/4/26 13:12
# Filename: vgg.py
import torch
import torch.nn as nn

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name: str, num_classes: int = 10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(num_classes: int = 10) -> nn.Module:
    return VGG('VGG11', num_classes=num_classes)


def vgg13(num_classes: int = 10) -> nn.Module:
    return VGG('VGG13', num_classes=num_classes)


def vgg16(num_classes: int = 10) -> nn.Module:
    return VGG('VGG16', num_classes=num_classes)


def vgg19(num_classes: int = 10) -> nn.Module:
    return VGG('VGG19', num_classes=num_classes)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = vgg16(num_classes=10)
    model = model.to(device)
    x = torch.randn(1, 3, 32, 32, device=device)
    y = model(x)
    print(y.size())

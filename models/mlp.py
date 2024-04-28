# -*- coding: utf-8 -*-
# Author  : liyanpeng
# Email   : yanpeng.li@cumt.edu.cn
# Datetime: 2024/4/26 13:39
# Filename: mlp.py
import torch
import torch.nn as nn

__all__ = ['mlp']


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def mlp(num_classes: int = 10) -> nn.Module:
    return MLP(num_classes)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mlp(num_classes=10)
    model = model.to(device)
    x = torch.randn(1, 3, 32, 32, device=device)
    y = model(x)
    print(y.size())

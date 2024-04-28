# -*- coding: utf-8 -*-
# Author  : liyanpeng
# Email   : yanpeng.li@cumt.edu.cn
# Datetime: 2024/3/30 14:07
# Filename: main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from models.mlp import mlp
from models.vgg import vgg16
from models.resnet import resnet50
from models.mobilenetv2 import mobilenet_v2

import os
import random
from datetime import datetime
import numpy as np
from loguru import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-mn', default='resnet50', type=str,
                    help='model name')
parser.add_argument('--opt_name', '-on', default='adam', type=str,
                    help='optimizer name')
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float,
                    help='learning rate')
args = parser.parse_args()

# # <<<<< 单机多卡
# import argparse
# import torch.distributed as dist
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', default=-1, type=int,
#                     help='node rank for distributed training')
# args = parser.parse_args()
#
# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)
# # >>>>>

seed = 10086
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

# time_str = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = args.model_name
opt_name = args.opt_name
learning_rate = args.learning_rate
time_str = '{}_{}_lr{}'.format(model_name, opt_name, learning_rate)
save_path = os.path.join('ckpts', time_str)
log_path = os.path.join('logs', time_str)
writer = SummaryWriter(log_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0
epochs = 100

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
# train_sampler = DistributedSampler(train_dataset)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, sampler=train_sampler)

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
# test_sampler = DistributedSampler(test_dataset)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4, sampler=test_sampler)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def select_model_and_opt(model_name: str = 'resnet50', opt_name: str = 'adam', learning_rate: float = 0.001):
    if model_name == 'mlp':
        model = mlp(num_classes=10)
    elif model_name == 'vgg16':
        model = vgg16(num_classes=10)
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(num_classes=10)
    else:
        model = resnet50(num_classes=10)
    model = model.to(device)
    if device == 'cuda':
        model = DataParallel(model, device_ids=[0, 1])
        cudnn.benchmark = True
    # model = DistributedDataParallel(model, device_ids=[args.local_rank])

    if opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer


criterion = nn.CrossEntropyLoss()
model, optimizer = select_model_and_opt(model_name, opt_name, learning_rate)
# scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)


# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        writer.add_scalar('loss/train', train_loss / (batch_idx + 1), epoch)
        writer.add_scalar('acc/train', correct / total, epoch)

    if epoch % 10 == 0:
        logger.info('[{}/{}] TrainAcc: {:.4f} | TrainLoss: {:.4f}'.format(epoch+1, epochs, correct / total, train_loss / len(train_loader)))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            writer.add_scalar('loss/test', test_loss / (batch_idx + 1), epoch)
            writer.add_scalar('acc/test', correct / total, epoch)

    if epoch % 10 == 0:
        logger.info('[{}/{}] TestAcc: {:.4f} | TestLoss: {:.4f}'.format(epoch+1, epochs, correct / total, test_loss / len(test_loader)))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        torch.save(model.state_dict(), os.path.join(save_path, '{}_{}.pth'.format(model_name, round(acc, 2))))
        best_acc = acc


for epoch in range(epochs):
    train(epoch)
    test(epoch)

print(best_acc)

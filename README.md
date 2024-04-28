# UCAS-DLWork
中国科学院大学人工智能学院课程项目作业【CIFAR10分类】

# Models
- MLP
- VGG系列（VGG11、VGG13、VGG16、VGG18）
- ResNet系列（ResNet18、ResNet34、ResNet50、ResNet101、ResNet152）
- MobileNetv2

# Datasets
- CIFAR10
- CIFAR100

# Train
```shell
# DP模式
CUDA_VISIBLE_DEVICES=0,1 python3 train.py

# DDP模式
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py
```

# Results
    显卡：V100*2 batchsize：512 seed：10086 epochs：100
|    `模型结构`     |   `优化器`   |  `学习率`   |  `准确率`   |
|:-------------:|:---------:|:--------:|:--------:|
|  `ResNet50`   |   `SGD`   |  `0.01`  | `0.8518` |
|  `ResNet50`   |   `SGD`   | `0.001`  | `0.6617` |
|  `ResNet50`   |   `SGD`   | `0.0001` | `0.3368` |
|  `ResNet50`   | `RMSprop` |  `0.01`  | `0.9128` |
|  `ResNet50`   | `RMSprop` | `0.001`  | `0.9184` |
|  `ResNet50`   | `RMSprop` | `0.0001` | `0.8971` |
|  `ResNet50`   |  `Adam`   |  `0.01`  | `0.9114` |
|  `ResNet50`   |  `Adam`   | `0.001`  | `0.9219` |
|  `ResNet50`   |  `Adam`   | `0.0001` | `0.8828` |
|  `ResNet18`   |  `Adam`   | `0.001`  | `0.9251` |
|     `MLP`     |  `Adam`   | `0.001`  | `0.5438` |
|    `VGG16`    |  `Adam`   | `0.001`  | `0.9141` |
| `MobileNetv2` |  `Adam`   | `0.001`  | `0.9035` |

# Reference
[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

#!/usr/bin/env bash

python3 train.py --model_name 'resnet50' --opt_name 'adam' -lr 0.01
python3 train.py --model_name 'resnet50' --opt_name 'adam' -lr 0.001
python3 train.py --model_name 'resnet50' --opt_name 'adam' -lr 0.0001
python3 train.py --model_name 'resnet50' --opt_name 'rmsprop' -lr 0.01
python3 train.py --model_name 'resnet50' --opt_name 'rmsprop' -lr 0.001
python3 train.py --model_name 'resnet50' --opt_name 'rmsprop' -lr 0.0001
python3 train.py --model_name 'resnet50' --opt_name 'sgd' -lr 0.01
python3 train.py --model_name 'resnet50' --opt_name 'sgd' -lr 0.001
python3 train.py --model_name 'resnet50' --opt_name 'sgd' -lr 0.0001
python3 train.py --model_name 'mlp' --opt_name 'adam' -lr 0.001
python3 train.py --model_name 'vgg16' --opt_name 'adam' -lr 0.001
python3 train.py --model_name 'mobilenet_v2' --opt_name 'adam' -lr 0.001
python3 train.py --model_name 'resnet18' --opt_name 'adam' -lr 0.001

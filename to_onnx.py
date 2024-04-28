# -*- coding: utf-8 -*-
# Author  : liyanpeng
# Email   : yanpeng.li@cumt.edu.cn
# Datetime: 2024/4/26 12:56
# Filename: to_onnx.py
import torch
from models.vgg import vgg16


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dummy_input = torch.randn(1, 3, 32, 32, device=device)
model = vgg16(num_classes=10)
model = model.to(device)

torch.onnx.export(model,
                  (dummy_input, ),
                  'vgg16.onnx',
                  input_names=["input"],
                  output_names=["output"])

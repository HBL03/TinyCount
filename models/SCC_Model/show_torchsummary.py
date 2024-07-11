import torch
import torch.nn as nn
from torchsummary import summary
from MyNet import MyNet

model = MyNet()
model = model.to('cuda')

input_size = (3, 768, 1024)

print(summary(model, input_size))
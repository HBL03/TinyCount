import sys
sys.path.append('C:\ML_practice\RA\Crowdcounting_Framework\MyNet-Test')

import torch
import torch.nn as nn
from torchsummary import summary
from SANet import SANet


model = SANet()
model = model.to('cuda')

input_size = (3, 768, 1024)

print(summary(model, input_size))
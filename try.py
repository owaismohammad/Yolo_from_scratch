from torch import batch_norm
import torch.nn as nn

def forward(self, x):
    layers =[]
    layers +=[
        self.conv(x),
        self.batchnorm(x),
        self.leakyrelu(x)
    ]
    return nn.Sequential(*layers)
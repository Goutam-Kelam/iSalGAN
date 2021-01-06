import torch.nn as nn
import torch
import math
from my_ops import *

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.convs = nn.Sequential( # [-1, 4, 300,300]
                conv2d(4,3,kernel_size=1),
                relu(),
                conv2d(3,32,kernel_size=3), #[-1, 32, 300,300]
                relu(),
                maxpool2d(),
                conv2d(32,64,kernel_size=3), #[-1, 64, 150,150]
                relu(),
                conv2d(64,64,kernel_size=3), #[-1, 64, 150,150]
                relu(),
                maxpool2d(), #[-1,64,75,75 ]
                conv2d(64,64,kernel_size=3), #[-1,64,75,75]
                relu(),
                conv2d(64,64,kernel_size=3),
                relu(),
                maxpool2d(), #[-1,64,37,37]
        )
        self.mymodules = nn.ModuleList([
            nn.Sequential(nn.Linear(64*37*37, 100), nn.Tanh()),
            nn.Sequential(nn.Linear(100,2), nn.Tanh()),
            nn.Sequential(nn.Linear(2,1), nn.Sigmoid())
        ])
        #self._initialize_weights()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.mymodules[0](x)
        x = self.mymodules[1](x)
        x = self.mymodules[2](x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

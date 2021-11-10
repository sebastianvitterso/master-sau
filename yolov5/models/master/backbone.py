from torch import nn
import torch
from torch.functional import Tensor

from models.common import C3, SPPF, Conv


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([
            Conv(3, 64, 6, 2, 2),
            Conv(64, 128, 3, 2),
            nn.Sequential(
                C3(128, 128),
                C3(128, 128),
                C3(128, 128),
            ),
            Conv(128, 256, 3, 2),
            nn.Sequential(
                C3(256, 256),
                C3(256, 256),
                C3(256, 256),
                C3(256, 256),
                C3(256, 256),
                C3(256, 256),
            ),
            Conv(256, 512, 3, 2),
            nn.Sequential(
                C3(512, 512),
                C3(512, 512),
                C3(512, 512),
                C3(512, 512),
                C3(512, 512),
                C3(512, 512),
                C3(512, 512),
                C3(512, 512),
                C3(512, 512),
            ),
            Conv(512, 1024, 3, 2),
            nn.Sequential(
                C3(1024, 1024),
                C3(1024, 1024),
                C3(1024, 1024),
            ),
            SPPF(1024, 1024, 5)
        ])

    def forward(self, x: Tensor):
        history = [None] * 25
        for i, module in enumerate(self.model):
            x = module(x)
            if i in [4,6]:
                history[i] = x.detach().clone()
        
        return x, history



if __name__ == '__main__':
    model = Backbone()
    x = torch.zeros(1, 3, 256, 256)
    y = model(x)
    print(y.results())
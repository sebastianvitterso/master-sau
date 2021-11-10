from torch import nn

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_rgb, x_ir):
        return x_rgb
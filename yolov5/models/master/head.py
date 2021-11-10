import math
from typing import List
from torch import nn
import torch
from torch.functional import Tensor

from models.common import C3, Concat, Conv
from models.yolo import Detect
from utils.autoanchor import check_anchor_order

class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.l10_conv = Conv(1024, 512, 1, 1)
        self.l11_upsample = nn.Upsample(None, 2, 'nearest')
        self.l12_concat = Concat(1)
        self.l13_c3 = nn.Sequential(
            C3(1024, 512),
            C3(512, 512),
            C3(512, 512),
        )

        self.l14_conv = Conv(512, 256, 1, 1)
        self.l15_upsample = nn.Upsample(None, 2, 'nearest')
        self.l16_concat = Concat(1)
        self.l17_c3 = nn.Sequential(
            C3(512, 256),
            C3(256, 256),
            C3(256, 256),
        )

        self.l18_conv = Conv(256, 256, 3, 2)
        self.l19_concat = Concat(1)
        self.l20_c3 = nn.Sequential(
            C3(512, 512),
            C3(512, 512),
            C3(512, 512),
        )

        self.l21_conv = Conv(512, 512, 3, 2)
        self.l22_concat = Concat(1)
        self.l23_c3 = nn.Sequential(
            C3(1024, 1024),
            C3(1024, 1024),
            C3(1024, 1024),
        )

        anchors = [
            [10,13, 16,30, 33,23],  # P3/8
            [30,61, 62,45, 59,119],  # P4/16
            [116,90, 156,198, 373,326],  # P5/32
        ]

        self.l24_detect = Detect(nc=1, anchors=anchors, ch=(512,512,512))
        # s = 256  # 2x min stride
        # self.l24_detect.inplace = True
        # self.l24_detect.stride = torch.tensor([8,16,32])
        # self.l24_detect.anchors /= self.l24_detect.stride.view(-1, 1, 1)
        # check_anchor_order(self.l24_detect)
        # for mi, s in zip(self.l24_detect.m, self.l24_detect.stride):  # from
        #     b = mi.bias.view(self.l24_detect.na, -1)  # conv.bias(255) to (3,85)
        #     b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        #     b.data[:, 5:] += math.log(0.6 / (self.l24_detect.nc - 0.99))
        #     mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)



    def forward(self, x: Tensor, history: List):
        x = self.l10_conv(x)
        history[10] = x.detach().clone()
        x = self.l11_upsample(x)
        x = self.l12_concat([x, history[6]])
        x = self.l13_c3(x)

        x = self.l14_conv(x)
        history[14] = x.detach().clone()
        x = self.l15_upsample(x)
        x = self.l16_concat([x, history[4]])
        x = self.l17_c3(x)
        history[17] = x.detach().clone()

        x = self.l18_conv(x)
        x = self.l19_concat([x, history[14]])
        x = self.l20_c3(x)
        history[20] = x.detach().clone()

        x = self.l21_conv(x)
        x = self.l22_concat([x, history[10]])
        x = self.l23_c3(x)
        history[23] = x.detach().clone()

        x = self.l24_detect([
            history[17],
            history[20],
            history[23],
        ])

        return x
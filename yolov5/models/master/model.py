from torch import nn

from models.master.backbone import Backbone
from models.master.fusion import Fusion
from models.master.head import Head

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone_rgb = Backbone()
        self.backbone_ir = Backbone()
        self.head = Head()
        self.fusion = Fusion()

        self.history_fusion_4 = Fusion()
        self.history_fusion_6 = Fusion()

    def forward(self, x_rgb, x_ir):
        x_rgb, history_rgb = self.backbone_rgb(x_rgb)
        x_ir, history_ir = self.backbone_ir(x_ir)

        x = self.fusion(x_rgb, x_ir)

        history = [None] * 25
        history[4] = self.history_fusion_4(history_rgb[4], history_ir[4])
        history[6] = self.history_fusion_6(history_rgb[6], history_ir[6])
         
        x = self.head(x, history)

        return x

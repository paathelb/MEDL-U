"""
    A customized Feature Pyramid Network.
"""

import torch
from torch import nn

class FPN(nn.Module):
    def __init__(self, in_channel=3, inter_size=128, out_channel=None) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(inter_size, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(inter_size, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(inter_size, inter_size, 3, padding=1),
        #     nn.BatchNorm2d(inter_size),
        #     nn.ReLU()
        # )
        self.raise_channel = nn.Conv2d(inter_size*3, out_channel, 1)
        
    def forward(self, img):
        # img: (B, C, H, W)
        x1 = self.layer1(img)  # B x C1 X H x W
        x2 = self.layer2(x1)   # B x C1 X H x W
        x3 = self.layer3(x2)   # B x C1 X H x W
        # x4 = self.layer4(x3)

        fpn = torch.cat([x1, x2, x3], dim=1)  # B x 3*C1 X H x W
        fpn = self.raise_channel(fpn)  # B x 512 X H x W
        return fpn, x3
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from registry import register_model
from torch.nn.functional import normalize
from .MobileIE_modules import MBRConv5, MBRConv3, MBRConv1,DropBlock,FST, FSTS


import matplotlib.pyplot as plt
from .ops import imshow_tensor, batch_masked_rgb_mean, awb

@register_model('MobileIE')
class MobileIENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(MobileIENet, self).__init__()
        self.channels = channels
        self.out_ch = 3
        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )



        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            # nn.Dropout2d(dropout_rate * 0.6)),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.att1 = nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )

        self.tail = nn.Sequential(
            nn.MaxPool2d(8, 8),
            MBRConv3(channels, 64, rep_scale=rep_scale),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(64, 3,1,1),
            nn.ReLU(inplace=True)
        )

        # "如果需要warmup，需修改这部分"
        # self.tail_warm = nn.Sequential(
        #     nn.MaxPool2d(8, 8),
        #     MBRConv3(channels, channels * 2, rep_scale=rep_scale),  # 增加通道
        #     nn.PReLU(channels * 2),
        #     nn.Dropout2d(0.1),
        #     nn.MaxPool2d(4, 4),  # 改为4x4，保留更多信息
        #     MBRConv3(channels * 2, channels, rep_scale=rep_scale),
        #     nn.PReLU(channels),
        #     nn.Dropout2d(0.15),
        #     nn.AdaptiveAvgPool2d(2),  # 2x2 instead of 1x1
        #     nn.Conv2d(channels, 3, 1, 1),
        #     nn.ReLU(inplace=True)
        # )



        # self.tail = nn.Sequential(
        #     nn.MaxPool2d(8, 8),  # 256->64
        #     MBRConv3(channels, 64, rep_scale=rep_scale),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(dropout_rate),
        #     nn.MaxPool2d(8, 8),  # 64->8
        #     nn.Conv2d(64, 3, 1, 1),  # 通道调整
        #     # nn.AdaptiveAvgPool2d(1),  # 8->1
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.tail_warm = nn.Sequential(
        #     nn.MaxPool2d(8, 8),  # 256->32
        #     MBRConv3(channels, 64, rep_scale=rep_scale),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(dropout_rate),
        #     nn.Conv2d(64, 3, 1, 1),
        #     # nn.AdaptiveAvgPool2d(1),  # 32->1
        #     nn.ReLU(inplace=True)
        # )
        #
        # "这里该怎么改呢？"
        # self.tail_warm = self.tail
        #
        # self.tail = nn.Sequential(nn.PixelShuffle(2),
        #                           MBRConv3(3, channels, rep_scale=rep_scale),
        #                           nn.AdaptiveAvgPool2d(1),
        #                           nn.ReLU(inplace=True)  )
        #
        # self.tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.drop = DropBlock(3)

    def forward(self, x):
        x0 = self.head(x)

        x1 = self.body(x0)

        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1

        out = self.tail(x4)
        pred = normalize(out.sum(-1).sum(-1), dim=1)
        return pred

    def forward_warm(self, x):
        ""
        "input: bz,3,256,256"
        x = self.drop(x)
        x = self.head(x)    ## bz,3,256,256 ---> bz, 12, 256, 256
        x = self.body(x)    ## bz, 12, 256, 256 ---> bz, 12, 256, 256


        "需要修改这里 --> ① 直接估计光照 ② 图像colorchecker平均值"
        out = self.tail(x)  ## bz, 12, 256, 256 ---> bz, 3, 256, 256
        # out_warm = self.tail_warm(x)   ## bz, 12, 256, 256 ---> bz, 3, 256, 256
        pred = normalize(out.sum(-1).sum(-1), dim=1)
        # pred_warm = normalize(out_warm.sum(-1).sum(-1), dim=1)
        return pred

    def slim(self):
        net_slim = MobileIEISPNetS(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5) or isinstance(mod, MBRConv1):
                if '%s.weight' % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim['%s.weight' % name] = w
                    weight_slim['%s.bias' % name] = b
            elif isinstance(mod, FST):
                weight_slim['%s.bias' % name] = mod.bias
                weight_slim['%s.weight1' % name] = mod.weight1
                weight_slim['%s.weight2' % name] = mod.weight2
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim


class MobileIEISPNetS(nn.Module):
    def __init__(self, channels):
        super(MobileIEISPNetS, self).__init__()
        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(4, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )
        self.body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.att1 = nn.Sequential(
            nn.Conv2d(1, channels, 1, 1),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1
        return self.tail(x4)

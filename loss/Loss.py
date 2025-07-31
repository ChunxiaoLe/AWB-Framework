import torch
from torch import Tensor
from torch.nn.functional import normalize
import math
import torch.nn as nn


class Loss:
    # def __init__(self):

    def _compute(self, *args, **kwargs) -> Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self._compute(*args)




class LossIllum(Loss):
    def __init__(self):
        super(LossIllum, self).__init__()
        "光照估计任务中，后续需要添加Loss的话，添加到这个class下"
        # self.loss_cs = nn.CosineSimilarity()
        self.loss_angular = AngularLoss()

    def _compute(self, out, gt):
        # loss = (1 - self.loss_cs(out, gt)).mean() + self.loss_angular(out,gt)

        loss =  self.loss_angular(out, gt)
        return loss





class AngularLoss(Loss):
    def __init__(self):
        super().__init__()

    def _compute(self, pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)





import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['angularloss', 'crossentropyloss']


class AngularLoss(nn.Module):
    def __init__(self, margin_a=1, margin_m=0,
                 margin_b=0, scale=64, gamma=0, **kwargs):
        super(AngularLoss, self).__init__()

        self.margin_a = margin_a
        self.margin_m = margin_m
        self.margin_b = margin_b
        self.scale = scale
        self.gamma = gamma
        self.m = torch.nn.LogSoftmax(dim=1)
        
        # initialization https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):

    def forward(self, input, target):
        zy = input * self.scale  # Batch * num_class
        gt_one_hot = zy.data * 0.0  # Batch * num_class
        gt_one_hot.scatter_(1, target.data.view(-1, 1), 1)
        gt_one_hot = gt_one_hot.byte()
        gt_one_hot = torch.autograd.Variable(gt_one_hot)

        sel_cos_t = torch.masked_select(zy, gt_one_hot)  # Batch (zy = mx.sym.pick(fc7, gt_label, axis=1))

        cos_value = sel_cos_t / self.scale  # (cos_t = zy/s)
        theta = torch.acos(cos_value)  # (t = mx.sym.arccos(cos_t))
        theta = theta * self.margin_a  # (t = t*args.margin_a)
        theta = theta + self.margin_m  # (t = t+args.margin_m)
        cos_t = torch.cos(theta)  # (body = mx.sym.cos(t))

        cos_t = cos_t - self.margin_b  # (body = body - args.margin_b)
        new_zy = cos_t * self.scale  # (new_zy = body*s)
        output = zy + torch.mul(gt_one_hot.float(), (new_zy - sel_cos_t).view(-1, 1))  # Batch * num_class
        # entropy
        # implementation 1
        # logpt = F.log_softmax(output, dim=1)
        # logpt = logpt.gather(1, target.view(-1, 1))
        # logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())

        # loss = -1 * (1-pt)**self.gamma * logpt
        # loss = loss.mean()

        # implementation 2 (same with 1)
        output = self.m(output)
        loss = F.nll_loss(output, target)
        return loss


def angularloss(margin_a=1, margin_m=0, margin_b=0, scale=64, gamma=0, **kwargs):
    layer = AngularLoss(margin_a, margin_m, margin_b, scale, gamma, **kwargs)
    return layer


def crossentropyloss(**kwargs):
    return torch.nn.CrossEntropyLoss()


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Linear']


class Linear(nn.Module):
    def __init__(self, num_embedding=512, num_class=1000, **kwargs):
        super(Linear, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_embedding, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        # self.fc = nn.Linear(num_embedding, num_class)
        # nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):

        embedding_unit = F.normalize(input, p=2, dim=1, eps=1e-12)
        # print('embedding_unit', embedding_unit.shape)
        weights_unit = F.normalize(self.weight, p=2, dim=0, eps=1e-12)
        # print('weights_unit', weights_unit.shape)
        cos_t = torch.mm(embedding_unit, weights_unit)  # Batch * num_class

        # x = input   # size=(B,F)    F is feature len
        # w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        # ww = w.renorm(2, 1, 1e-5).mul(1e5)
        # xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        # wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        # cos_t = x.mm(ww)  # size=(B,Classnum)
        # cos_t = cos_t / xlen.view(-1, 1) / wlen.view(1, -1)
        # cos_t = cos_t.clamp(-1, 1)

        # cos_t = self.fc(input)

        return cos_t


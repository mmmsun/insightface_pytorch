import torch.nn as nn


__all__ = ['embeddingE']


class EmbeddingE(nn.Module):

    def __init__(self, inplanes, input_size, num_embedding, dp, **kwargs):

        super(EmbeddingE, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, affine=False, eps=2e-05, momentum=0.9)
        self.dp = nn.Dropout(p=dp)
        self.fc = nn.Linear(input_size, num_embedding)
        self.bn2 = nn.BatchNorm1d(num_embedding, affine=True, eps=2e-05, momentum=0.9)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        return x


def embeddingE(inplanes, input_size, num_embedding, dp, **kwargs):
    layer = EmbeddingE(inplanes, input_size, num_embedding, dp, **kwargs)
    return layer

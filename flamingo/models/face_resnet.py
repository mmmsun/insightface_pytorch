import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['FaceResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet100',
           'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockV3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlockV3, self).__init__()
        act_type = kwargs.get('version_act', 'prelu')
        self.use_se = kwargs.get('version_se', 1)
        self.bn_mom = kwargs.get('bn_mom', 0.9)

        self.bn1 = nn.BatchNorm2d(inplanes, affine=False, eps=2e-5, momentum=self.bn_mom)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, eps=2e-5, momentum=self.bn_mom)
        if act_type == 'relu':
            self.act1 = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            self.act1 = nn.PReLU()
        else:
            raise ValueError('not valid version_act: {}'.format(act_type))

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes, affine=False, eps=2e-5, momentum=self.bn_mom)

        if self.use_se:
            self.se_avgpool = nn.AdaptiveAvgPool2d(1)
            self.se_conv1 = nn.Conv2d(planes, planes//16, kernel_size=1, stride=1, padding=0)

            if act_type == 'relu':
                self.se_act = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                self.se_act = nn.PReLU()
            else:
                raise ValueError('not valid version_act: {}'.format(act_type))

            self.se_conv2 = nn.Conv2d(planes//16, planes, kernel_size=1, stride=1, padding=0)
            self.se_sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.act1(out)
        out = self.conv2(out)

        out = self.bn3(out)

        if self.use_se:
            out_se = self.se_avgpool(out)
            out_se = self.se_conv1(out_se)
            out_se = self.se_act(out_se)
            out_se = self.se_conv2(out_se)
            out_se = self.se_sigmoid(out_se)

            out = out_se*out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class BottleneckV3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BottleneckV3, self).__init__()
        act_type = kwargs.get('version_act', 'prelu')
        self.use_se = kwargs.get('version_se', 1)
        self.bn_mom = kwargs.get('bn_mom', 0.9)

        self.bn1 = nn.BatchNorm2d(inplanes, affine=False, eps=2e-5, momentum=self.bn_mom)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, eps=2e-5, momentum=self.bn_mom)
        if act_type == 'relu':
            self.act1 = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            self.act1 = nn.PReLU()
        else:
            raise ValueError('not valid version_act: {}'.format(act_type))

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes, affine=False, eps=2e-5, momentum=self.bn_mom)
        if act_type == 'relu':
            self.act2 = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            self.act2 = nn.PReLU()
        else:
            raise ValueError('not valid version_act: {}'.format(act_type))

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * self.expansion, affine=False, eps=2e-5, momentum=self.bn_mom)

        if self.use_se:
            self.se_conv1 = nn.Conv2d(inplanes, planes//self.expansion, kernel_size=1, stride=1, padding=0, bias=False)

            if act_type == 'relu':
                self.se_act = nn.ReLU(inplace=True)
            else:
                self.se_act = nn.PReLU()

            self.se_conv2 = nn.Conv2d(planes//self.expansion, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.se_sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn4(out)

        if self.use_se:
            out_se = nn.GovalAvgPooling(out)
            out_se = self.se_conv1(out_se)
            out_se = self.se_act(out_se)
            out_se = self.se_conv2(out_se)
            out_se = self.se_sigmoid(out_se)
            out = out_se*out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FaceResNetEmbedding(nn.Module):

    def __init__(self, block, layers, embedding_block, **kwargs):
        self.bn_mom = kwargs.get('bn_mom', 0.9)

        self.inplanes = 64
        super(FaceResNetEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5, momentum=self.bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)
        # self.embedding = self._make_embedding_layer(512*7*7, num_embedding, **kwargs)
        self.embedding = embedding_block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2.0))
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=False, eps=2e-5, momentum=self.bn_mom),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.embedding(x)
        return x


class FaceResNet(nn.Module):

    def __init__(self, block, layers, embedding_block, classifier_block, **kwargs):
        self.inplanes = 64
        super(FaceResNet, self).__init__()
        self.net = FaceResNetEmbedding(block, layers, embedding_block, **kwargs)
        self.classifier = classifier_block

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FaceResNet(BasicBlockV3, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError('no pretrained model exists')
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FaceResNet(BasicBlockV3, [3, 4, 6, 3], **kwargs)
    if pretrained:
        raise ValueError('no pretrained model exists')
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FaceResNet(BasicBlockV3, [3, 4, 14, 3], **kwargs)
    if pretrained:
        raise ValueError('no pretrained model exists')
    return model


def resnet100(pretrained=False, **kwargs):
    """Constructs a ResNet-100 model.

    """
    model = FaceResNet(BasicBlockV3, [3, 13, 30, 3], **kwargs)
    if pretrained:
        raise ValueError('no pretrained model exists')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FaceResNet(BottleneckV3, [3, 4, 23, 3], **kwargs)
    if pretrained:
        raise ValueError('no pretrained model exists')
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FaceResNet(BottleneckV3, [3, 8, 36, 3], **kwargs)
    if pretrained:
        raise ValueError('no pretrained model exists')    
    return model

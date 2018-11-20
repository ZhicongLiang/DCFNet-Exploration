'''
Ref: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo
from models.DCF import Conv_DCF

k = -1
weights = np.load('/home/zliangak/zhicongliang/DCFNet-VGG/bases/bases_resnet.npy')

def CONV_DCF(in_channels, out_channels, kernel_size, num_bases, initializer,
             stride=1, padding=0, bias=True):
    global k
    if initializer == 'PCA':
        k+=1
        return Conv_DCF(in_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride, padding=padding, num_bases=num_bases,
                        bias = bias, initializer = weights[k])
    else:
        
        return Conv_DCF(in_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride, padding=padding, num_bases=num_bases,
                        bias = bias, initializer = initializer)
        



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_bases, stride=1 ,downsample=None, initializer='FB'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CONV_DCF(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False, num_bases=num_bases ,initializer=initializer)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # if the channels of x and out is different
        # then we need to downsample x to cater out
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers ,in_channels, num_bases, initializer ,data_expansion=1 , num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                               bias=False)
        self.conv1 = CONV_DCF(in_channels, 64, kernel_size=7, stride=2, padding=3, 
                              bias=False, num_bases=num_bases, initializer=initializer)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,layers[0], num_bases=num_bases,initializer=initializer)
        self.layer2 = self._make_layer(block, 128, layers[1], num_bases=num_bases, stride=2, initializer=initializer)
        self.layer3 = self._make_layer(block, 256, layers[2], num_bases=num_bases, stride=2, initializer=initializer)
        self.layer4 = self._make_layer(block, 512, layers[3], num_bases=num_bases, stride=2, initializer=initializer)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion * data_expansion**2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks , num_bases, initializer,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, 
                            stride=stride, 
                            downsample=downsample,
                            num_bases=num_bases, 
                            initializer=initializer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_bases=num_bases, initializer=initializer))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model